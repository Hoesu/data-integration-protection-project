from src.database.insert import insert_card_user_info_safe

if __name__ == "__main__":
    insert_card_user_info_safe(
        csv_path="/home/hoesu.chung/GITHUB/project/data/sample.csv",
        create_table=True
    )