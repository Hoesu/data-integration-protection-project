from src.database import *


if __name__ == "__main__":
    
    engine = Base.get_engine()
    Base.metadata.create_all(engine)
    
    insert_data(
        engine=engine,
        csv_path="/home/hoesu.chung/GITHUB/project/data/201807_회원정보.csv",
        target_table=CardUserInfo,
        batch_size=1000,
    )