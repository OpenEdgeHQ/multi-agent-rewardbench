import aiohttp
import re
from trafilatura import load_html, baseline, fetch_url
from trafilatura.settings import BASIC_CLEAN_XPATH
from trafilatura.xml import delete_element
from lxml.html import HtmlElement


def clean_text(text):
    text = text.strip()
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text


def clean_html(html):
    tree = load_html(html)
    body = tree.find(".//body")
    def basic_cleaning(tree: HtmlElement) -> HtmlElement:
        for elem in BASIC_CLEAN_XPATH(tree):
            delete_element(elem)
        return tree
    body = basic_cleaning(body)
    return clean_text(body.text_content())


async def fetch(fetch_url: str) -> str:
    """
    异步获取网页内容
    """
    base_url = 'https://app.scrapingbee.com/api/v1'
    api_key = 'xxx'

    configs = [
        {
            'api_key': api_key,
            'url': fetch_url,
            'country_code': 'us',
            'block_resources': 'false'
        },
        {
            'api_key': api_key,
            'url': fetch_url,
            'premium_proxy': 'true',
            'country_code': 'us',
            'block_resources': 'false'
        },
        {
            'api_key': api_key,
            'url': fetch_url,
            'stealth_proxy': 'true',
            'country_code': 'us',
            'block_resources': 'false'
        }
    ]

    async with aiohttp.ClientSession() as session:
        for config in configs:
            try:
                async with session.get(base_url, params=config) as response:
                    if response.status == 200:
                        return await response.text()
            except aiohttp.ClientError:
                continue

    return ""
    # print('Response HTTP Status Code: ', response.status_code)
    # print('Response HTTP Response Body: ', response.content)


if __name__ == '__main__':
    import asyncio

    url = "https://zhuanlan.zhihu.com/p/1925858930858386901"
    url = "https://movie.douban.com/subject/1291832/"
    asyncio.run(fetch(url))
