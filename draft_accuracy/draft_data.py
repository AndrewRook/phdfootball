import requests
import sys
import lxml.html as lh
from lxml.etree import tostring
import pandas as pd

def download_draft_data(start_year, end_year):
    data = pd.concat([
        _download_one_year_data(year)
        for year in range(start_year, end_year + 1)
    ])
    data = data.reset_index()
    return data

def _download_one_year_data(year):
    print(f"downloading {year} draft")
    page = lh.document_fromstring(
        requests.get(f"https://www.pro-football-reference.com/years/{year}/draft.htm").text
    )
    data = pd.read_html(tostring(page), attrs={"id": "drafts"})[0]
    data.columns = data.columns.droplevel()
    data = data.loc[data["Player"] != "Player", :].reset_index()

    base_player_url_xpath = "//table[@id='drafts']/tbody/tr/td[@data-stat='player']/"
    player_url_xpath_suffix_1 = "strong/a/@href"
    player_url_xpath_suffix_2 = "a/@href"
    data["player_url"] = page.xpath(
        f"{base_player_url_xpath}{player_url_xpath_suffix_1}|{base_player_url_xpath}{player_url_xpath_suffix_2}")
    return data
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Syntax: python download_draft_data.py [START_YEAR] [END_YEAR]")
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    data = download_draft_data(start_year, end_year)
    data.to_csv("data/draft_data.csv", index=False)
