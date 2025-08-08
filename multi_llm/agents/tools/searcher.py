import json
import requests

async def search(query):
    response = requests.get(
        url='https://app.scrapingbee.com/api/v1/store/google',
        params={
            'api_key': '1M2O3Q87TQ47ZA3JI25HNBFCEOF93P86D76EEOIDOH1C3TFNCTRKD31X1E8BROW3WPOYUXYAQUHNKI3O',
            'search': query,
            'language': 'en',
            'nb_results': '20',
            'page': '1',
            'add_html': 'true'
        },
    )
    # print('Response HTTP Status Code: ', response.status_code)
    # print('Response HTTP Response Body: ', response.content)
    if response.status_code != 200:
        return ""
    else:
        return response.text

if __name__ == '__main__':
    import asyncio
    res = asyncio.run(search("dune part 2"))
