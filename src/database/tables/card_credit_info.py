from datetime import date
from typing import Optional

from sqlalchemy import Date, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CardCreditInfo(Base):
    __tablename__ = "CARD_CREDIT_INFO"

    기준년월: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    발급회원번호: Mapped[str] = mapped_column(String, primary_key=True)
    최초한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    카드이용한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    CA한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    일시상환론한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    월상환론한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    CA이자율_할인전: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CL이자율_할인전: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    RV일시불이자율_할인전: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    RV현금서비스이자율_할인전: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    RV신청일자: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    RV약정청구율: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    RV최소결제비율: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    자발한도감액횟수_R12M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    자발한도감액금액_R12M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    자발한도감액후경과월: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    강제한도감액횟수_R12M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    강제한도감액금액_R12M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    강제한도감액후경과월: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도증액횟수_R12M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도증액금액_R12M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도증액후경과월: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    상향가능한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    상향가능CA한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    카드론동의여부: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    월상환론상향가능한도금액: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    RV전환가능여부: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    일시불ONLY전환가능여부: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    카드이용한도금액_B1M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    카드이용한도금액_B2M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    특별한도보유여부_R3M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    연체감액여부_R3M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도심사요청건수: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도요청거절건수: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도심사요청후경과월: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    한도심사거절후경과월: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    시장단기연체여부_R6M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    시장단기연체여부_R3M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    시장연체상환여부_R6M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    시장연체상환여부_R3M: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rv최초시작후경과일: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"CardCreditInfo(기준년월={self.기준년월!r}, 발급회원번호={self.발급회원번호!r})"
