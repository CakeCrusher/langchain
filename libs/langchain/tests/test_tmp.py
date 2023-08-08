from oplangchain.chains.openai_functions.openapi import get_openapi_chain


def test_tmp() -> None:
    chain = get_openapi_chain(
        "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
    )
    res = chain.run("What are some options for a men's large blue button down shirt")
    # assert that res object includes key products
    assert "products" in res
