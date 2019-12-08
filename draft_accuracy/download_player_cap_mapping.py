import json
import pandas as pd
import requests


if __name__ == "__main__":
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
    parsed_data.to_csv("data/cap_urls_mapping.csv", index=False)
    

