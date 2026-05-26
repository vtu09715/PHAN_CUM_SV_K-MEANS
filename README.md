# PHAN_CUM_SV_K-MEANS

# VŨ ĐỨC TÚ 


# LỚP K58.KTP

# MSV: K225480106068

# link video: https://youtu.be/q79TrHhgKhk?si=NGLabPlIgxAmxRJ8
# CODE CHƯƠNG TRÌNH 

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_excel("bangdiem.xlsx", header=None, keep_default_na=False)

# Tìm dòng chứa từ khóa
def tim_dong(tu_khoa):
    mask = df.apply(lambda row: row.astype(str).str.contains(tu_khoa, case=False).any(), axis=1)
    return mask[mask].index[0]

dong_mssv = tim_dong("MSSV")
dong_ten = tim_dong("Tên Sinh Viên")
dong_mon = tim_dong("Tên Môn học")

cot_bd = df.iloc[dong_mssv].astype(str).tolist().index("MSSV") + 1

mssv = df.iloc[dong_mssv, cot_bd:].reset_index(drop=True)
ten_sv = df.iloc[dong_ten, cot_bd:].reset_index(drop=True)
bang_diem = df.iloc[dong_mon + 1:, cot_bd:].reset_index(drop=True)

# Chuẩn hóa điểm
def chuan_hoa(x):
    x = str(x).strip().replace(",", ".")

    if x in ["NaN", "nAn", "—"]:
        return np.nan

    if x == "" or x.lower() == "x":
        return round(np.random.uniform(2.0, 4.0), 1)

    try:
        return float(x)
    except:
        return np.nan

bang_diem = bang_diem.map(chuan_hoa)

# Tính điểm trung bình hệ 4
diem_tb = bang_diem.mean(axis=0, skipna=True)

# Phân cụm K-Means
du_lieu = diem_tb.values.reshape(-1, 1)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
nhom = model.fit_predict(du_lieu)

# Đặt tên cụm theo điểm tăng dần
thu_tu = np.argsort(model.cluster_centers_.flatten())
ten_cum = dict(zip(thu_tu, ["Khá", "Giỏi", "Xuất sắc"]))

ket_qua = pd.DataFrame({
    "MSSV": mssv,
    "Tên sinh viên": ten_sv,
    "Điểm TK(4)": np.round(diem_tb.values, 2),
    "Cụm": [ten_cum[i] for i in nhom]
})

# Tách từng cụm thành các nhóm cột riêng
ds_nhom = [
    ket_qua[ket_qua["Cụm"] == loai][["MSSV", "Tên sinh viên", "Điểm TK(4)"]]
    .reset_index(drop=True)
    .add_prefix(f"{loai} - ")
    for loai in ["Khá", "Giỏi", "Xuất sắc"]
]

bang_xuat = pd.concat(ds_nhom, axis=1)
bang_xuat.to_excel("ketqua_phanCum.xlsx", index=False)

print("Đã xuất file ketqua_phanCum.xlsx đúng định dạng")




