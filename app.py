import streamlit as st
import pandas as pd
import numpy as np
from colour import Lab_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_Lab

# CSV ファイルの読み込み
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Lab → RGB 変換
def lab_to_rgb(lab):
    xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(
        xyz,
        illuminant_XYZ="D65",
        illuminant_RGB="D65",
        chromatic_adaptation_transform="Bradford",
        colourspace=RGB_COLOURSPACES["sRGB"],
    )
    return tuple(max(0, min(255, int(x * 255))) for x in rgb)

# RGB → Lab 変換
def rgb_to_lab(rgb):
    rgb_scaled = [x / 255.0 for x in rgb]
    xyz = RGB_to_XYZ(
        rgb_scaled,
        illuminant_RGB="D65",
        illuminant_XYZ="D65",
        chromatic_adaptation_transform="Bradford",
        colourspace=RGB_COLOURSPACES["sRGB"],
    )
    return XYZ_to_Lab(xyz)

# 独自の CIEDE2000 計算関数
def ciede2000(lab1, lab2, k_L=1, k_C=1, k_H=1):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    delta_L = L2 - L1
    L_ = (L1 + L2) / 2

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_ = (C1 + C2) / 2

    G = 0.5 * (1 - np.sqrt(C_**7 / (C_**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    delta_C_prime = C2_prime - C1_prime

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    delta_h_prime = 0
    if C1_prime * C2_prime != 0:
        if abs(h2_prime - h1_prime) <= 180:
            delta_h_prime = h2_prime - h1_prime
        elif h2_prime - h1_prime > 180:
            delta_h_prime = h2_prime - h1_prime - 360
        else:
            delta_h_prime = h2_prime - h1_prime + 360

    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime) / 2)

    h_bar_prime = h1_prime + h2_prime
    if abs(h1_prime - h2_prime) > 180:
        h_bar_prime += 360
    h_bar_prime /= 2

    T = (
        1
        - 0.17 * np.cos(np.radians(h_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3 * h_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    )

    delta_theta = 30 * np.exp(-((h_bar_prime - 275) / 25) ** 2)
    R_C = 2 * np.sqrt(C_**7 / (C_**7 + 25**7))
    S_L = 1 + ((0.015 * (L_ - 50) ** 2) / np.sqrt(20 + (L_ - 50) ** 2))
    S_C = 1 + 0.045 * C_
    S_H = 1 + 0.015 * C_ * T
    R_T = -np.sin(2 * np.radians(delta_theta)) * R_C

    delta_E = np.sqrt(
    (delta_L / (k_L * S_L))**2
    + (delta_C_prime / (k_C * S_C))**2
    + (delta_H_prime / (k_H * S_H))**2
    + R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
)
    return delta_E

# アプリケーションのレイアウト
st.title("CIEDE2000 色差検索アプリ")
st.sidebar.header("設定")

# CSV ファイルのアップロード
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)

    # 必須列のチェック
    required_columns = ["収録MS", "商品コード", "銘柄名", "色名", "L", "a", "b"]
    if not all(col in data.columns for col in required_columns):
        st.error(f"CSV ファイルには以下の列が必要です: {', '.join(required_columns)}")
    else:
        # キーワード検索とサジェスト
        st.sidebar.subheader("銘柄名・色名で検索")
        keyword = st.sidebar.text_input("キーワードを入力")
        selected_lab = None
        if keyword:
            filtered_data = data[
                data["銘柄名"].str.contains(keyword, na=False, case=False)
                | data["色名"].str.contains(keyword, na=False, case=False)
            ]
            if not filtered_data.empty:
                selected_row = st.sidebar.selectbox(
                    "候補", 
                    filtered_data.apply(
                        lambda row: f"{row['銘柄名']} - {row['色名']} (L={row['L']}, a={row['a']}, b={row['b']})",
                        axis=1
                    )
                )
                selected_index = filtered_data.index[
                    filtered_data.apply(
                        lambda row: f"{row['銘柄名']} - {row['色名']} (L={row['L']}, a={row['a']}, b={row['b']})",
                        axis=1
                    ) == selected_row
                ][0]
                selected_lab = filtered_data.loc[selected_index, ["L", "a", "b"]].values
                st.sidebar.write(f"選択された Lab 値: {selected_lab}")

        # KL 値としきい値の調整
        k_l = st.sidebar.slider("KL (明度の重み、値が小さいほど重みが増す)", 0.01, 3.0, 1.0, 0.01)
        threshold = st.sidebar.slider("色差のしきい値", 0.1, 20.0, 5.0, 0.1)

        # カラーピッカーと Lab 値入力
        st.sidebar.subheader("Lab 値を指定")
        st.sidebar.write("カラーピッカーで色を選択")
        selected_color = st.sidebar.color_picker("色を選択", "#808080")
        rgb_selected = tuple(int(selected_color[i:i+2], 16) for i in (1, 3, 5))
        lab_from_picker = rgb_to_lab(rgb_selected)

        input_l = st.sidebar.slider("L 値", 0.0, 100.0, selected_lab[0] if selected_lab is not None else round(lab_from_picker[0], 2), 0.1)
        input_a = st.sidebar.slider("a 値", -128.0, 128.0, selected_lab[1] if selected_lab is not None else round(lab_from_picker[1], 2), 0.1)
        input_b = st.sidebar.slider("b 値", -128.0, 128.0, selected_lab[2] if selected_lab is not None else round(lab_from_picker[2], 2), 0.1)
        user_lab = [input_l, input_a, input_b]

        # ユーザー入力のカラーチップ
        st.sidebar.subheader("入力された色のカラーチップ")
        user_rgb = lab_to_rgb(user_lab)
        user_color = f"rgb{user_rgb}"
        st.sidebar.markdown(
            f"<div style='width:50px; height:50px; background-color:{user_color};'></div>",
            unsafe_allow_html=True,
        )

        # 色差計算
        st.subheader("検索結果")
        results = []
        for _, row in data.iterrows():
            color_lab = [row["L"], row["a"], row["b"]]
            delta_e = ciede2000(user_lab, color_lab, k_L=k_l)
            if delta_e <= threshold:
                delta_l = round(color_lab[0] - user_lab[0], 2)
                delta_a = round(color_lab[1] - user_lab[1], 2)
                delta_b = round(color_lab[2] - user_lab[2], 2)
                color_rgb = lab_to_rgb(color_lab)
                results.append((
                    f"<div style='width:20px; height:20px; background-color:rgb{color_rgb};'></div>",
                    row["収録MS"],
                    row["商品コード"],
                    row["銘柄名"],
                    row["色名"],
                    round(row["L"], 2), round(row["a"], 2), round(row["b"], 2),
                    delta_l, delta_a, delta_b, round(delta_e, 2)
                ))

        # 結果を表示
        st.write(f"色差が {threshold} 以下の結果: {len(results)} 件")
        if results:
            result_df = pd.DataFrame(results, columns=[
                "カラーチップ", "収録MS", "商品コード", "銘柄名", "色名", "L", "a", "b", "ΔL", "Δa", "Δb", "色差"
            ])
            st.write(result_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("一致する色が見つかりませんでした。")

else:
    st.write("CSV ファイルをアップロードしてください。")
