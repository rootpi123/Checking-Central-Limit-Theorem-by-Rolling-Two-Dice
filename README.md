# Roll-Two-Dice
```
import random
import os
import time
from PIL import Image
from IPython.display import display, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, base64

sums = []

def get_valid_number_of_rolls():  # 사용자에게 유효한 주사위 던지기 횟수를 입력받는 함수
    while True:
        try:
            n = int(input("주사위를 던질 횟수를 입력하세요: "))
            if n > 0:
                return n
            else:
                print("양의 정수만 입력하세요.")
        except ValueError:
            print("양의 정수만 입력하세요.")

def get_valid_number_of_interval():  # 사용자에게 시행 간 시간 간격을 입력받는 함수
    while True:
        try:
            t = float(input("각 시행 사이의 간격을 입력하세요 (단위: 초): "))
            if t >= 0:
                return t
            else:
                print("0 이상의 수만 입력하세요.")
        except ValueError:
            print("0 이상의 수만 입력하세요.")

def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode()
    return "data:image/png;base64," + base64_str

def roll_and_display_dice(i):  # 주사위를 굴리고, 이미지를 표시하고, 히스토그램을 생성하는 함수
    # 주사위 두 개 굴리기
    dice1_roll = random.randint(1, 6)
    dice2_roll = random.randint(1, 6)
    dice_sum = dice1_roll + dice2_roll
    sums.append(dice_sum)

    img1_filename = f"dice_{dice1_roll}.png"
    img2_filename = f"dice_{dice2_roll}.png"

    # 이미지 파일 존재 확인
    if not (os.path.exists(img1_filename) and os.path.exists(img2_filename)):
        print("주사위 이미지가 없습니다. dice_1.png ~ dice_6.png 파일을 같은 폴더에 넣어주세요.")
        return
        
    # 이미지 파일 열기
    try:
        img1 = Image.open(img1_filename)
        img2 = Image.open(img2_filename)
    except Exception as e:
        print(f"이미지 열기 실패: {e}")
        return

    img1_data = pil_to_base64(img1)
    img2_data = pil_to_base64(img2)

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.25, 0.25, 0.5],
        subplot_titles=(f"Dice 1: {dice1_roll}", f"Dice 2: {dice2_roll}", "Sum Histogram"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )

    # 이미지 표시
    fig.add_layout_image(
        dict(source=img1_data, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1),
        row=1, col=1
    )
    fig.add_layout_image(
        dict(source=img2_data, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1),
        row=1, col=2
    )

    for col in [1, 2]:
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=col)
        fig.update_yaxes(visible=False, range=[0, 1], row=1, col=col)

    # 히스토그램 생성
    hist = go.Histogram(
        x=sums,
        xbins=dict(start=1.5, end=12.5, size=1),  # 막대를 2~12 정수 중심에 배치
        marker=dict(color='rgba(0, 100, 255, 0.7)', line=dict(width=1, color='black'))
    )
    fig.add_trace(hist, row=1, col=3)

    fig.update_xaxes(
        range=[1.5, 12.5],
        tickvals=list(range(2, 13)),
        ticktext=[str(i) for i in range(2, 13)],
        row=1, col=3
    )

    fig.update_layout(
        height=450,
        width=1000,
        title_text=f"Dice Roll Result - Attempt {i+1}/{n}",
        showlegend=False,
        bargap=0,
        bargroupgap=0
    )

    clear_output(wait=True)  # 이전 출력 삭제
    display(fig)             # 새 그래프 출력
    time.sleep(t)            # 간격 대기

if __name__ == "__main__":
    n = get_valid_number_of_rolls()    # 던질 횟수 입력 받기
    t = get_valid_number_of_interval() # 간격 입력 받기
    for i in range(n):                 # 주사위 굴리기
        roll_and_display_dice(i)
```
