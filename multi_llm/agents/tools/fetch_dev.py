url0 = "https://zhuanlan.zhihu.com/p/1925858930858386901"
url1 = "https://x.com/ziru999/status/1942481057976312040"
url2 = "https://www.reddit.com/r/webscraping/comments/1016j3l/best_web_scraping_apis_at_the_moment/"
# from zenrows import ZenRowsClient
#
# client = ZenRowsClient("6b1e752b75287f02796f985d94d981316a0013ef")

# url = url1
# response = client.get(url)
#
# print(response.text)


# import requests
# import urllib.parse
# token = "76a1efea3877430ea453dbbd2b2a45219e9850cf3f7"
# targetUrl = urllib.parse.quote(url1)
# url = "http://api.scrape.do/?token={}&url={}".format(token, targetUrl)
# response = requests.request("GET", url)
# print(response.text)


# Install the Python ScrapingBee library:
# pip install scrapingbee


#  Install the Python Requests library:
# `pip install requests`

#  Install the Python Requests library:
# `pip install requests`
import requests


def send_request():
    response = requests.get(
        url='https://app.scrapingbee.com/api/v1',
        params={
            'api_key': 'XXX',
            'url': 'https://zhuanlan.zhihu.com/p/1925858930858386901',
            'premium_proxy': 'true',
            'country_code': 'us',
            'block_resources': 'false'
        },
    )
    print('Response HTTP Status Code: ', response.status_code)
    print('Response HTTP Response Body: ', response.content)


send_request()

# def send_request():
#     response = requests.get(
#         url='https://app.scrapingbee.com/api/v1/store/google',
#         params={
#             'api_key': 'NK79A781O9QJTZI11KM23DHPFCEWKAOJCT0YTN6A234XWGXVVGLB4XLFENGJLT5VPEM9WSN04DORF9UH',
#             'search': 'web scraping API',
#             'language': 'en',
#             'nb_results': '10',
#             'add_html': 'false',
#             'page': 1
#         },
#     )
#     print('Response HTTP Status Code: ', response.status_code)
#     print('Response HTTP Response Body: ', response.content)


# send_request()
