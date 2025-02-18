# 2025-01-31,13点14 爬虫
'''
https://www.biccamera.com/bc/category/001/100/150/125/005/?sort=01
https://www.amazon.co.jp/dp/B0DTSN2K86?th=1&psc=1&m=AN1VRQENFRJN5&tag=twm1abf-22&linkCode=ogi
https://www.amazon.co.jp/dp/B0DV9BSDSM?th=1&psc=1&m=AN1VRQENFRJN5&tag=twm1abf-22&linkCode=ogi
https://www.amazon.co.jp/dp/B0DS2Z8854?th=1&psc=1&m=AN1VRQENFRJN5&tag=twm1abf-22&linkCode=ogi
https://www.amazon.co.jp/dp/B0DT7GMXHB?th=1&psc=1&m=AN1VRQENFRJN5&tag=twm1abf-22&linkCode=ogi
https://www.amazon.co.jp/dp/B0DT6Q3BXM?th=1&psc=1&m=AN1VRQENFRJN5&tag=twm1abf-22&linkCode=ogi


'''



url='https://www.amazon.co.jp/dp/B0DTSN2K86?th=1&psc=1&m=AN1VRQENFRJN5&tag=twm1abf-22&linkCode=ogi'


from bs4 import BeautifulSoup 



import requests
# a=requests.get(url)
a=requests.get('https://www.google.com').text
soup = BeautifulSoup(a, 'lxml')
print(a)








