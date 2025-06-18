# 중심 극한 정리(central limit theorem)
서로 독립이고 동일한 분포를 가지는 확률 변수가 n개 존재한다고 하자. n이 충분히 크다고 하면, n개의 평균의 분포는 정규분포에 가까워진다.
## 확인 방법
주사위 2개를 굴리는 시행을 통해 중심 극한 정리를 확인할 것이다.
확률 변수 X의 값을 두 주사위의 합으로 정의한다. 그러면, 각 시행의 결과는 다른 시행의 결과에 영향을 미치지 못하므로 모든 확률 변수는 독립이다. 또한, 모두 동일한 분포를 가진다.
S = X1 + X2 + ... + Xn으로 정의한다. 그렇다면 중심 극한 정리에 따라 n이 충분히 크다면 S의 평균의 분포는 정규 분포와 가까워질 것이다.
## 실행 과정
### 모듈 불러오기
먼저 코드 실행에 필요한 모듈들을 불러온다.
```python
import random                                      # 주사위 숫자 생성을 위한 랜덤 모듈
import os                                          # 파일 존재 여부 확인
import time                                        # 딜레이를 위한 시간 모듈
from PIL import Image                              # 이미지 처리용 라이브러리
from IPython.display import display, clear_output  # Jupyter에서 이미지/그래프 갱신
import plotly.graph_objects as go                  # Plotly 시각화 구성요소
from plotly.subplots import make_subplots          # 서브플롯 생성
import io, base64                                  # 이미지 인코딩을 위한 모듈
import numpy as np                                 # 수치 계산용
from scipy.stats import norm                       # 정규 분포 함수
```
### 시행 횟수(+시행 간 간격) 입력 함수 
다음으로는 주사위 2개를 굴리는 시행을 몇 번 반복할 것인지를 사용자에게 입력받는다. 시행 횟수는 양의 정수만 받을 수 있도록 한다.
추가로 각 시행 간의 간격 또한 입력받는다(이는 꼭 필요한 것은 아님).
```python
def get_valid_number_of_rolls():  # 사용자로부터 유효한 주사위 던지기 횟수를 입력받음
    while True:
        try:
            n = int(input("주사위를 던질 횟수를 입력하세요: "))
            if n > 0:
                return n
            else:           # 양이 아닌 정수를 입력한 경우
                print("양의 정수만 입력하세요.")
        except ValueError:  # 정수가 아닌 값을 입력한 경우
            print("양의 정수만 입력하세요.")

def get_valid_number_of_interval():  # 사용자로부터 유효한 간격(초)을 입력받음
    while True:
        try:
            t = float(input("각 시행 사이의 간격을 입력하세요 (단위: 초): "))
            if t >= 0:
                return t
            else:           # 음수를 입력한 경우
                print("0 이상의 수만 입력하세요.")
        except ValueError:  # 수가 아닌 값을 입력한 경우
            print("0 이상의 수만 입력하세요.")
```
### 시행 반복 및 결과 출력 함수
입력받은 값에 따라 주사위를 굴리는 시행을 반복하고 결과를 출력한다. 출력할 결과는 실제 주사위 눈(이미지), 두 주사위의 합(히스토그램), 정규 분포 곡선, 총 세 가지이다.
```python
sums = []  # 주사위 눈금 합을 저장할 리스트

def pil_to_base64(img):  # PIL 이미지 객체를 base64 문자열로 변환 (웹에서 표시 가능하게 함)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode()
    return "data:image/png;base64," + base64_str

def roll_and_display_dice(i):  # 주사위를 굴리고 결과를 시각화하는 함수

    # 주사위 두 개를 굴려 결과를 저장
    dice1_roll = random.randint(1, 6)
    dice2_roll = random.randint(1, 6)
    dice_sum = dice1_roll + dice2_roll
    sums.append(dice_sum)

    # 주사위 눈금에 해당하는 이미지 파일 이름 생성
    img1_filename = f"dice_{dice1_roll}.png"
    img2_filename = f"dice_{dice2_roll}.png"

    # 이미지 파일 열기
    try:
        img1 = Image.open(img1_filename)
        img2 = Image.open(img2_filename)
    except Exception as e:
        print(f"이미지 열기 실패: {e}")
        return

    # 이미지 base64로 인코딩
    img1_data = pil_to_base64(img1)
    img2_data = pil_to_base64(img2)

    # Plotly 서브플롯 구성: 주사위 2개 + 히스토그램
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.25, 0.25, 0.5],
        subplot_titles=(f"Dice 1: {dice1_roll}", f"Dice 2: {dice2_roll}", "Sum Histogram"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )

    # 첫 번째 주사위 이미지 추가
    fig.add_layout_image(
        dict(source=img1_data, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1),
        row=1, col=1
    )

    # 두 번째 주사위 이미지 추가
    fig.add_layout_image(
        dict(source=img2_data, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1),
        row=1, col=2
    )

    # 이미지 축 감추기
    for col in [1, 2]:
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=col)
        fig.update_yaxes(visible=False, range=[0, 1], row=1, col=col)

    # 주사위 합 히스토그램 생성
    hist = go.Histogram(
        x=sums,
        xbins=dict(start=1.5, end=12.5, size=1),
        marker=dict(color='rgba(0, 100, 255, 0.7)', line=dict(width=1, color='black')),
        histnorm='probability'  # 확률로 정규화된 히스토그램
    )
    fig.add_trace(hist, row=1, col=3)

    # 2개 이상 시행 시 정규분포 곡선 추가
    if len(sums) >= 2:
        mu = np.mean(sums)
        sigma = np.std(sums)
        x_vals = np.linspace(2, 12, 200)
        y_vals = norm.pdf(x_vals, mu, sigma)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='red', width=2),
                name='Normal Distribution'
            ),
            row=1, col=3
        )

    # 히스토그램 축 설정
    fig.update_xaxes(
        range=[1.5, 12.5],
        tickvals=list(range(2, 13)),
        ticktext=[str(i) for i in range(2, 13)],
        row=1, col=3
    )

    # 전체 레이아웃 설정
    fig.update_layout(
        height=450,
        width=1000,
        title_text=f"Dice Roll Result - Attempt {i+1}/{n}",
        showlegend=False,
        bargap=0,
        bargroupgap=0,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # 이전 출력 지우고 새로 출력
    clear_output(wait=True)
    display(fig)

    # 다음 시행까지 대기
    time.sleep(t)  
```
### 파일 존재 여부 확인 및 최종 실행
마지막으로 주사위 눈의 이미지 파일이 존재하는 지 확인한다. 존재하지 않을 경우 그 사실을 알리며 실행을 멈춘다. 존재할 경우 함수들을 실행한다.
```python
if __name__ == "__main__":
    # 필요한 주사위 이미지 파일이 모두 존재하는지 확인
    missing_images = [f"dice_{i}.png" for i in range(1, 7) if not os.path.exists(f"dice_{i}.png")]
    if missing_images:
        print("다음 주사위 이미지 파일이 누락되었습니다:")
        for fname in missing_images:
            print(f" - {fname}")
        print("dice_1.png ~ dice_6.png 파일을 같은 폴더에 넣어주세요.")

    else:
        # 입력 함수 실행
        n = get_valid_number_of_rolls()
        t = get_valid_number_of_interval()
        # 입력한 값만큼 반복 실행
        for i in range(n):
            roll_and_display_dice(i)
```
## 결과 확인
실제로 중심 극한 정리가 작동하는 지 확인한다. 시행 횟수를 늘려가며 히스토그램이 정규 분포 곡선에 가까워지는 것을 확인할 수 있다.
### n = 10
![Image](https://github.com/user-attachments/assets/e6b69162-b859-4e8d-82fd-b531b1137fe0)
### n = 100

### n = 1000


## 전체 코드
```python
import random                                      # 주사위 숫자 생성을 위한 랜덤 모듈
import os                                          # 파일 존재 여부 확인
import time                                        # 딜레이를 위한 시간 모듈
from PIL import Image                              # 이미지 처리용 라이브러리
from IPython.display import display, clear_output  # Jupyter에서 이미지/그래프 갱신
import plotly.graph_objects as go                  # Plotly 시각화 구성요소
from plotly.subplots import make_subplots          # 서브플롯 생성
import io, base64                                  # 이미지 인코딩을 위한 모듈
import numpy as np                                 # 수치 계산용
from scipy.stats import norm                       # 정규 분포 함수

def get_valid_number_of_rolls():  # 사용자로부터 유효한 주사위 던지기 횟수를 입력받음
    while True:
        try:
            n = int(input("주사위를 던질 횟수를 입력하세요: "))
            if n > 0:
                return n
            else:           # 양이 아닌 정수를 입력한 경우
                print("양의 정수만 입력하세요.")
        except ValueError:  # 정수가 아닌 값을 입력한 경우
            print("양의 정수만 입력하세요.")

def get_valid_number_of_interval():  # 사용자로부터 유효한 간격(초)을 입력받음
    while True:
        try:
            t = float(input("각 시행 사이의 간격을 입력하세요 (단위: 초): "))
            if t >= 0:
                return t
            else:           # 음수를 입력한 경우
                print("0 이상의 수만 입력하세요.")
        except ValueError:  # 수가 아닌 값을 입력한 경우
            print("0 이상의 수만 입력하세요.")

sums = []  # 주사위 눈금 합을 저장할 리스트

def pil_to_base64(img):  # PIL 이미지 객체를 base64 문자열로 변환 (웹에서 표시 가능하게 함)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode()
    return "data:image/png;base64," + base64_str

def roll_and_display_dice(i):  # 주사위를 굴리고 결과를 시각화하는 함수

    # 주사위 두 개를 굴려 결과를 저장
    dice1_roll = random.randint(1, 6)
    dice2_roll = random.randint(1, 6)
    dice_sum = dice1_roll + dice2_roll
    sums.append(dice_sum)

    # 주사위 눈금에 해당하는 이미지 파일 이름 생성
    img1_filename = f"dice_{dice1_roll}.png"
    img2_filename = f"dice_{dice2_roll}.png"

    # 이미지 파일 열기
    try:
        img1 = Image.open(img1_filename)
        img2 = Image.open(img2_filename)
    except Exception as e:
        print(f"이미지 열기 실패: {e}")
        return

    # 이미지 base64로 인코딩
    img1_data = pil_to_base64(img1)
    img2_data = pil_to_base64(img2)

    # Plotly 서브플롯 구성: 주사위 2개 + 히스토그램
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.25, 0.25, 0.5],
        subplot_titles=(f"Dice 1: {dice1_roll}", f"Dice 2: {dice2_roll}", "Sum Histogram"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )

    # 첫 번째 주사위 이미지 추가
    fig.add_layout_image(
        dict(source=img1_data, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1),
        row=1, col=1
    )

    # 두 번째 주사위 이미지 추가
    fig.add_layout_image(
        dict(source=img2_data, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1),
        row=1, col=2
    )

    # 이미지 축 감추기
    for col in [1, 2]:
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=col)
        fig.update_yaxes(visible=False, range=[0, 1], row=1, col=col)

    # 주사위 합 히스토그램 생성
    hist = go.Histogram(
        x=sums,
        xbins=dict(start=1.5, end=12.5, size=1),
        marker=dict(color='rgba(0, 100, 255, 0.7)', line=dict(width=1, color='black')),
        histnorm='probability'  # 확률로 정규화된 히스토그램
    )
    fig.add_trace(hist, row=1, col=3)

    # 2개 이상 시행 시 정규분포 곡선 추가
    if len(sums) >= 2:
        mu = np.mean(sums)
        sigma = np.std(sums)
        x_vals = np.linspace(2, 12, 200)
        y_vals = norm.pdf(x_vals, mu, sigma)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='red', width=2),
                name='Normal Distribution'
            ),
            row=1, col=3
        )

    # 히스토그램 축 설정
    fig.update_xaxes(
        range=[1.5, 12.5],
        tickvals=list(range(2, 13)),
        ticktext=[str(i) for i in range(2, 13)],
        row=1, col=3
    )

    # 전체 레이아웃 설정
    fig.update_layout(
        height=450,
        width=1000,
        title_text=f"Dice Roll Result - Attempt {i+1}/{n}",
        showlegend=False,
        bargap=0,
        bargroupgap=0,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # 이전 출력 지우고 새로 출력
    clear_output(wait=True)
    display(fig)

    # 다음 시행까지 대기
    time.sleep(t)

if __name__ == "__main__":
    # 필요한 주사위 이미지 파일이 모두 존재하는지 확인
    missing_images = [f"dice_{i}.png" for i in range(1, 7) if not os.path.exists(f"dice_{i}.png")]
    if missing_images:
        print("다음 주사위 이미지 파일이 누락되었습니다:")
        for fname in missing_images:
            print(f" - {fname}")
        print("dice_1.png ~ dice_6.png 파일을 같은 폴더에 넣어주세요.")

    else:
        # 입력 함수 실행
        n = get_valid_number_of_rolls()
        t = get_valid_number_of_interval()
        # 입력한 값만큼 반복 실행
        for i in range(n):
            roll_and_display_dice(i)
```
