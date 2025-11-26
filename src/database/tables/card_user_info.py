from datetime import date

from sqlalchemy import Date, Integer, String
from sqlalchemy.event import listens_for
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CardUserInfo(Base):
    __tablename__ = 'CARD_USER_INFO'

    기준년월: Mapped[date | None] = mapped_column(Date, nullable=True)
    발급회원번호: Mapped[str] = mapped_column(String, primary_key=True)
    남녀구분코드: Mapped[int | None] = mapped_column(Integer, nullable=True)
    연령: Mapped[str | None] = mapped_column(String(10), nullable=True)
    VIP등급코드: Mapped[str | None] = mapped_column(String(10), nullable=True)
    최상위카드등급코드: Mapped[str | None] = mapped_column(String(10), nullable=True)
    회원여부_이용가능: Mapped[int | None] = mapped_column(Integer, nullable=True)
    회원여부_이용가능_CA: Mapped[int | None] = mapped_column(Integer, nullable=True)
    회원여부_이용가능_카드론: Mapped[int | None] = mapped_column(Integer, nullable=True)
    소지여부_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    소지카드수_유효_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    소지카드수_이용가능_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    입회일자_신용: Mapped[date | None] = mapped_column(Date, nullable=True)
    입회경과개월수_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    회원여부_연체: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용거절여부_카드론: Mapped[int | None] = mapped_column(Integer, nullable=True)
    동의여부_한도증액안내: Mapped[int | None] = mapped_column(Integer, nullable=True)
    수신거부여부_TM: Mapped[int | None] = mapped_column(Integer, nullable=True)
    수신거부여부_DM: Mapped[int | None] = mapped_column(Integer, nullable=True)
    수신거부여부_메일: Mapped[int | None] = mapped_column(Integer, nullable=True)
    수신거부여부_SMS: Mapped[int | None] = mapped_column(Integer, nullable=True)
    가입통신회사코드: Mapped[str | None] = mapped_column(String(10), nullable=True)
    탈회횟수_누적: Mapped[int | None] = mapped_column(Integer, nullable=True)
    최종탈회후경과월: Mapped[int | None] = mapped_column(Integer, nullable=True)
    탈회횟수_발급6개월이내: Mapped[int | None] = mapped_column(Integer, nullable=True)
    탈회횟수_발급1년이내: Mapped[int | None] = mapped_column(Integer, nullable=True)
    거주시도명: Mapped[str | None] = mapped_column(String(20), nullable=True)
    직장시도명: Mapped[str | None] = mapped_column(String(20), nullable=True)
    마케팅동의여부: Mapped[int | None] = mapped_column(Integer, nullable=True)
    유효카드수_신용체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    유효카드수_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    유효카드수_신용_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    유효카드수_체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    유효카드수_체크_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능카드수_신용체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능카드수_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능카드수_신용_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능카드수_체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능카드수_체크_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용카드수_신용체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용카드수_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용카드수_신용_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용카드수_체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용카드수_체크_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용금액_R3M_신용체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용금액_R3M_신용: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용금액_R3M_신용_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용금액_R3M_체크: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용금액_R3M_체크_가족: Mapped[int | None] = mapped_column(Integer, nullable=True)
    _1순위카드이용금액: Mapped[int | None] = mapped_column(Integer, nullable=True)
    _1순위카드이용건수: Mapped[int | None] = mapped_column(Integer, nullable=True)
    _1순위신용체크구분: Mapped[str | None] = mapped_column(String(10), nullable=True)
    _2순위카드이용금액: Mapped[int | None] = mapped_column(Integer, nullable=True)
    _2순위카드이용건수: Mapped[int | None] = mapped_column(Integer, nullable=True)
    _2순위신용체크구분: Mapped[str | None] = mapped_column(String(10), nullable=True)
    최종유효년월_신용_이용가능: Mapped[date | None] = mapped_column(Date, nullable=True)
    최종유효년월_신용_이용: Mapped[date | None] = mapped_column(Date, nullable=True)
    최종카드발급일자: Mapped[date | None] = mapped_column(Date, nullable=True)
    보유여부_해외겸용_본인: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능여부_해외겸용_본인: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용여부_3M_해외겸용_본인: Mapped[int | None] = mapped_column(Integer, nullable=True)
    보유여부_해외겸용_신용_본인: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용가능여부_해외겸용_신용_본인: Mapped[int | None] = mapped_column(Integer, nullable=True)
    이용여부_3M_해외겸용_신용_본인: Mapped[int | None] = mapped_column(Integer, nullable=True)
    연회비발생카드수_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    연회비할인카드수_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    기본연회비_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    제휴연회비_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    할인금액_기본연회비_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    할인금액_제휴연회비_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    청구금액_기본연회비_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    청구금액_제휴연회비_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    상품관련면제카드수_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    임직원면제카드수_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    우수회원면제카드수_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    기타면제카드수_B0M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    카드신청건수: Mapped[int | None] = mapped_column(Integer, nullable=True)
    Life_Stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    최종카드발급경과월: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f'CardUserInfo(발급회원번호={self.발급회원번호!r}, 기준년월={self.기준년월!r})'


@listens_for(CardUserInfo, 'before_insert', propagate=True)
@listens_for(CardUserInfo, 'before_update', propagate=True)
def preprocess(mapper, connection, target):
    if target.발급회원번호 is None:
        raise ValueError('발급회원번호는 필수 필드입니다.')

    for column in mapper.columns:
        value = getattr(target, column.name, None)
        if column.name in ['기준년월', '최종유효년월_신용_이용가능', '최종유효년월_신용_이용']:
            setattr(target, column.name, Base._normalize_yyyymm_date(value))
        elif column.name in ['입회일자_신용', '최종카드발급일자']:
            setattr(target, column.name, Base._normalize_yyyymmdd_date(value))
        else:
            setattr(target, column.name, Base._normalize_missing_value(value))


@listens_for(CardUserInfo, 'load', propagate=True)
def postprocess(target, context):
    if target.연령 is not None:
        target.연령 = target.연령.replace('대', '')
        target.연령 = target.연령.replace('이상', '')
        target.연령 = int(target.연령)
    if target.Life_Stage is not None:
        target.Life_Stage = int(target.Life_Stage.split('.')[0])
