import bs4
import urllib.request as url

job = input("Enter the job : ")
path = "https://www.indeed.co.in/jobs?q={}&l=".format(job)
res = url.urlopen(path)

page = bs4.BeautifulSoup(res,'lxml')
jobTitle = page.find_all('a',class_='jobtitle')
location = page.find_all('div',class_='sjcl')

for i in range(len(jobTitle)):
    print(jobTitle[i].text)
    l = " ".join(location[i].text.split())
    print(l)
