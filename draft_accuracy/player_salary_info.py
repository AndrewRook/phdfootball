import json
import pandas as pd
import requests


def download_player_salary_urls():
    raw_data = json.loads(
        requests.get(
            "https://overthecap.com/Includes/ajax/players.php"
        ).text
    )
    parsed_data = pd.DataFrame(raw_data)
    parsed_data["url"] = (
        "https://overthecap.com/player/aaaaa/" +
        parsed_data["value"].astype(str)
    )
    return parsed_data

def download_player_salary_data(player_salary_url):
    data = pd.read_html(player_salary_url, match="Contract Type")
    return data[0]

if __name__ == "__main__":
    parsed_data = download_player_salary_urls()
    parsed_data.to_csv("data/salary_urls_mapping.csv", index=False)
    

