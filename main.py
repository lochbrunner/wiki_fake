#!/usr/bin/env python

import scrapy


class QuotesSpider(scrapy.Spider):
    name = 'wiki'

    def start_requests(self):
        urls = ['https://de.wikipedia.org/wiki/Wikipedia:Hauptseite']
        for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        print(response.url)
