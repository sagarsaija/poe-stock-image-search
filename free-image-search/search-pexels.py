import requests
import argparse


def search_pexels_images(api_key, query, per_page):
    url = f'https://api.pexels.com/v1/search?query={query}&per_page={per_page}'
    headers = {
        'Authorization': api_key
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    image_links = []
    for photo in data['photos']:
        image_links.append(photo['src']['large'])

    return image_links


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search images from Pexels')
    parser.add_argument('api_key', help='Pexels API key')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--per-page', type=int, default=10,
                        help='Number of images per page (default: 10)')

    args = parser.parse_args()

    image_links = search_pexels_images(args.api_key, args.query, args.per_page)

    print(f'Image links for "{args.query}":')
    for link in image_links:
        print(link)
