from oplangchain.chains.llm import LLMChain
from oplangchain.chat_models.openai import ChatOpenAI
from oplangchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from oplangchain.prompts.chat import ChatPromptTemplate
from oplangchain.chains.openai_functions.openapi import get_openapi_chain
from oplangchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from oplangchain.utilities.openapi import OpenAPISpec
from typing import Union
import json


test_success_plugins = [
    {
        "name": "BrowserOp",
        "openapi_url": "https://testplugin.feednews.com/.well-known/openapi.yaml",
        "messages": [
            {
                "role": "user",
                "content": "hi what is https://chat.openai.com/",
            }
        ],
        "truncate": False,
    },
    {
        "name": "BrowserPilot",
        "openapi_url": "https://browserplugin.feednews.com/.well-known/openapi.yaml",
        "messages": [
            {
                "role": "user",
                "content": "hi what is https://chat.openai.com/",
            }
        ],
        "truncate": False,
    },
    {
        "name": "twtData",
        "openapi_url": "https://www.twtdata.com/openapi.yaml",
        "messages": [
            {
                "role": "user",
                "content": "show me the amount of people @Sebasti54919704 is following",
            }
        ],
        "truncate": False,
    }
]
test_root_url_fail_plugins = [
    {
        "name": "askyourpdf",
        "openapi_url": "https://plugin.askyourpdf.com/.well-known/openapi.yaml",
        "messages": [
            {
                "role": "user",
                "content": "summarize this pdf https://eforms.com/download/2018/01/Non-Disclosure-Agreement-Template.pdf",
            }
        ],
        "truncate": False,
    },
    {
        "name": "make_an_excel_sheet",
        "openapi_url": "https://sheet-generator.brandzzy.com/openapi.yaml",
        "messages": [
            {
                "role": "user",
                "content": "Create a CSV that has one column of fake names make 10 of them",
            }
        ],
        "truncate": False,
    },
    {
        "name": "andorra_news_flats_traffic_work__search",
        "openapi_url": "https://gpt.andocarbur.com/openai.yaml",
        "messages": [
            {
                "role": "user",
                "content": "andorra traffic work",
            }
        ],
        "truncate": False,
    },
    {
        "name": "MermaidChart",
        "openapi_url": "https://www.mermaidchart.com/chatgpt/openapi.json",
        "messages": [
            {
                "role": "user",
                "content": "Create a mermaid diagram whhere a car node connects to 4 wheel nodes",
            }
        ],
        "truncate": False,
    },
]
test_plugin = test_success_plugins[2]


def test_full_suite() -> None:
    def openapi_to_functions_and_call_api_fn():
        openapi_url = test_plugin["openapi_url"]
        print(f"\"{test_plugin['name']}\" openapi_url: ", openapi_url)
        if openapi_url == None:
            raise ValueError("OpenAPI URL not found in manifest")
        if isinstance(openapi_url, Union[OpenAPISpec, str]):
            for conversion in (
                # each of the below specs can get stuck in a while loop
                # TODO: only OpenAPISpec.from_url is needed for OpenPlugin
                OpenAPISpec.from_url,
                OpenAPISpec.from_file,
                OpenAPISpec.from_text,
            ):
                try:
                    print("trying conversion: ", openapi_url)
                    openapi_url = conversion(openapi_url)  # type: ignore[arg-type]
                    break
                except Exception:  # noqa: E722
                    pass
            if isinstance(openapi_url, str):
                raise ValueError(f"Unable to parse spec from source {openapi_url}")
        openai_fns, call_api_fn = openapi_spec_to_openai_fn(openapi_url)
        print(
            f"\"{test_plugin['name']}\" functions: ", json.dumps(openai_fns, indent=2)
        )
        return openai_fns, call_api_fn

    openai_fns, call_api_fn = openapi_to_functions_and_call_api_fn()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template("{query}"),
        llm_kwargs={"functions": openai_fns},
        output_parser=JsonOutputFunctionsParser(args_only=False),
        output_key="function",
        verbose=True,
        # **(llm_kwargs or {}),
    )

    def estimate_tokens(s: str) -> int:
        return len(s) // 2

    def tokens_to_chars(tokens: int) -> int:
        return tokens * 2

    functions_tokens = estimate_tokens(json.dumps(openai_fns))

    try:
        # MESSAGES TO PROMPT
        # if there is a message with role system then pop it, iterate through all messages to find it
        system_message = ""
        for message in test_plugin["messages"]:
            if message["role"] == "system":
                system_message = "system" + ": " + message["content"] + "\n"
                test_plugin["messages"].remove(message)
                break

        # print("system_message: ", system_message)
        # Combine messages into one string
        messages_aggregate = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in test_plugin["messages"]
            ]
        )
        complete_messages_aggregate_tokens = estimate_tokens(
            system_message + messages_aggregate
        )
        # print("complete_messages_aggregate_tokens: ", complete_messages_aggregate_tokens)
        # print("functions_tokens: ", functions_tokens)
        messages_truncation_offset = tokens_to_chars(
            max(complete_messages_aggregate_tokens + functions_tokens - 4096, 0)
        )
        # print("messages_truncation_offset: ", messages_truncation_offset)
        messages_aggregate = messages_aggregate[messages_truncation_offset:]

        # TODO: temp fix to prevent collation of messages
        if messages_truncation_offset > 0:
            messages_aggregate = "user/assistant: " + messages_aggregate

        complete_messages_aggregate = system_message + messages_aggregate
        # print("complete_messages_aggregate: ", complete_messages_aggregate)
        # print("final length: ", estimate_tokens(complete_messages_aggregate))

        # Replace prompt with messageAggregate
        llm_chain_out = llm_chain.run(complete_messages_aggregate)
        print("Using plugin: " + test_plugin["name"])
    except KeyError as e:
        # if error includes "function_call" then it is not a plugin function
        if "function_call" in str(e):
            print("returned1:", str(e))
            raise ValueError("Not a plugin function")
        else:
            raise e
    if llm_chain_out["name"] not in [function["name"] for function in openai_fns]:
        print("returned2:", json.dumps(llm_chain_out, indent=2))
        raise ValueError("Not a plugin function")

    # EDGE CASE
    def remove_empty_from_dict(input_dict):
        cleaned_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                v = remove_empty_from_dict(v)
            if v and v != "none":  # only add to cleaned_dict if v is not empty
                cleaned_dict[k] = v
        return cleaned_dict

    llm_chain_out["arguments"] = remove_empty_from_dict(llm_chain_out["arguments"])
    print(
        f"\"{test_plugin['name']}\" llm_chain_out: ",
        json.dumps(llm_chain_out, indent=2),
    )

    # make the api call
    def request_chain(name, arguments):
        print(
            "request_chain name: {}, arguments: {}".format(name, json.dumps(arguments))
        )
        res = call_api_fn(name, arguments, headers=None, params=None)
        return res

    request_out = request_chain(**llm_chain_out)
    print("request_out status code: ", request_out.status_code)
    # print("request_out content: ", request_out.text)
    json_response = request_out.json()

    def truncate_json_root(json_response, truncate_to):
        return json_response

    if test_plugin["truncate"]:
        truncate_to = (
            test_plugin["truncate"]
            if not isinstance(test_plugin["truncate"], bool)
            else None
        )
        if truncate_to is None:
            token_slack = 56 + 300
            truncate_to = (
                4096
                - estimate_tokens(json.dumps(test_plugin["messages"][-1]))
                - token_slack
                - 0
            )
        json_response = truncate_json_root(json_response, truncate_to)

    print(
        f"\"{test_plugin['name']}\" json_response: ",
        json.dumps(json_response, indent=2),
    )
    assert 1 == 2
    try:
        return {
            "role": "function",
            "name": llm_chain_out["name"],
            "content": json.dumps(json_response),
        }
    except json.decoder.JSONDecodeError:
        raise json.decoder.JSONDecodeError(
            f"API call failed, API returned the following non-JSON response:\n{response.content}"
        )
