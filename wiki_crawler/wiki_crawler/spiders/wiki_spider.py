
import scrapy
import html2text


def invalid(s):
    return '(' in s or ')' in s or '[' in s or ']' in s or '/' in s


class QuotesSpider(scrapy.Spider):
    name = 'wiki'

    allowed_domains = ['de.wikipedia.org']
    start_urls = ['http://de.wikipedia.org/wiki/Mathematik']

    def __init__(self):
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True

    def parse(self, response):
        paragraphs = response.css('#bodyContent p').getall()

        for paragraph in paragraphs:
            for sentence in self.converter.handle(paragraph).split('. '):
                if len(sentence) > 0:
                    if invalid(sentence):
                        continue

                    yield {'sentence': sentence.replace('\n', ' ')}

        links = response.css('#bodyContent a::attr(href)').getall()
        for link in links:
            yield response.follow(link, callback=self.parse)
