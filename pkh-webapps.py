import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.naive_bayes import GaussianNB

st.write("""
# Sistem Rekomendasi Ekspor Untuk UMKM""")

img = Image.open('ekspor.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

modelnb = pickle.load(open('./Model/modelNBC_PKHv2.pkl', 'rb'))


def run():
    u = {25: 'Biji kakao',
         24: 'Minyak kelapa sawit dan fraksinya, dimurnikan maupun tidak (tidak termasuk yang dimodifikasi secara kimia)',
         23: 'Kain tenunan dari sutera atau dari sisa sutera',
         22: 'Produk tekstil dan barang untuk keperluan teknis',
         21: 'Kulit kecokelatan atau kerak dan kulit sapi termasuk. binatang kerbau atau kuda yang tidak berbulu',
         20: 'Peralatan makan, peralatan dapur, perlengkapan rumah tangga lainnya, dan perlengkapan toilet dari porselen atau keramik',
         19: 'Peralatan makan, peralatan dapur, perlengkapan rumah tangga lainnya, dan perlengkapan toilet dari plastik',
         18: 'Makanan olahan diperoleh dengan cara menggembungkan atau memanggang sereal atau produk sereal',
         17: 'Anyaman dan produk semacam itu dari bahan anyaman dirakit menjadi anyaman strip maupun tidak',
         16: 'Kertas dan kertas karton lainnya tidak dilapisi dalam gulungan dengan lebar 36 cm atau berbentuk persegi atau persegi panjang',
         15: 'Barang lainnya dari kayu',
         14: 'Asesoris pakaian jadi dan bagian dari garmen atau segala jenis asesoris pakaian',
         13: 'Perhiasan imitasi',
         12: 'Alas kaki dengan sol luar dari kulit karet plastik atau kulit komposisi dan bagian atasnya',
         11: 'Bumbu makanan',
         10: 'Kopi, digongseng atau dihilangkan kafeinnya, sekam dan kulit kopi pengganti kopi',
         9: 'Barang keramik',
         8: 'Sediaan kecantikan atau tata rias dan sediaan perawatan kulit termasuk tabir surya',
         7: 'Pakaian dan aksesoris pakaian dari kulit atau kulit komposisi',
         6: 'Pupuk nitrogen mineral atau kimia (tidak termasuk yang berbentuk tablet atau bentuk serupa',
         5: 'Serat optik dan bundel serat optik Kabel serat optik',
         4: 'Setelan jas pria atau anak laki-laki, jaket, blazer, turunannya',
         3: 'Barang dari plastik dan produk turunannya',
         2: 'Gandum dan Meslin, Biji-Bijian yang Mengandung Karbohidrat',
         1: 'Briket/Bahan Bakar padat serupa yang dibuat dari batu bara'}
    us = list(u.keys())
    usia = st.selectbox('Jenis Barang', us, format_func=lambda x: u[x])

    j = {3: '< 2', 2: '<=2', 1: '> 2'}
    jm = list(j.keys())
    jml_tgg = st.selectbox('JUMLAH_TANGGUNGAN', jm, format_func=lambda x: j[x])

    p = {7: 'PELAJAR/MAHASISWA', 6: 'MENGURUS RUMAH TANGGA', 5: 'BURUH LEPAS', 4: 'BURUH TANI',
         3: 'PETANI/PEKEBUN', 2: 'KARYAWAN SWASTA', 1: 'WIRASWASTA', 0: 'GURU/PNS/TNI/POLRI'}
    pk = list(p.keys())
    pekerjaan = st.selectbox('PEKERJAAN', pk, format_func=lambda x: p[x])

    ph = {4: 'TIDAK BERPENGHASILAN', 3: '< 1.000.000',
          2: '1.000.000 - 2.000.000', 1: '> 2.000.000'}
    phs = list(ph.keys())
    penghasilan = st.selectbox('PENGHASILAN', phs, format_func=lambda x: ph[x])

    t = {3: 'KONTRAK', 2: 'BEBAS SEWA', 1: 'MILIK SENDIRI'}
    tg = list(t.keys())
    tggl = st.selectbox('TEMPAT_TINGGAL', tg, format_func=lambda x: t[x])

    jl = {3: 'TANAH', 2: 'SEMEN', 1: 'KERAMIK'}
    jlt = list(jl.keys())
    jn_lt = st.selectbox('JENIS_LANTAI', jlt, format_func=lambda x: jl[x])

    jd = {3: 'ANYAMAN BAMBU', 2: 'KAYU', 1: 'SEMEN'}
    jdd = list(jd.keys())
    jn_dd = st.selectbox('JENIS_DINDING', jdd, format_func=lambda x: jd[x])

    b = {2: 'TIDAK', 1: 'YA'}
    bl = list(b.keys())
    bntl = st.selectbox('BANTUAN_LAIN', bl, format_func=lambda x: b[x])

    st.subheader('Tabel Inputan Data')
    data = {'USIA': usia,
            'JUMLAH_TANGGUNGAN': jml_tgg,
            'PEKERJAAN': pekerjaan,
            'PENGHASILAN': penghasilan,
            'TEMPAT_TINGGAL': tggl,
            'JENIS_LANTAI': jn_lt,
            'JENIS_DINDING': jn_dd,
            'BANTUAN_LAIN': bntl, }
    fitur = pd.DataFrame(data, index=[0])
    st.write(fitur)

    #prediksi = modelnb.predict(fitur)
    pred_prob = modelnb.predict_proba(fitur)

   # st.subheader('Keterangan Label Kelas')

   # st.write(keterangan)

    st.subheader('Hasil Prediksi (Klasifikasi Penerima Bantuan PKH)')
    if st.button("Submit"):
        fitur = pd.DataFrame(data, index=[0])
        print(fitur)
        prediction = modelnb.predict(fitur)
        keterangan = np.array(['LAYAK', 'TIDAK LAYAK'])
        lc = [str(i) for i in prediction]
        keterangan = np.array(0)
        labels2 = (prediction[keterangan])
        if keterangan == 0:
            st.error(
                labels2
            )
        else:
            st.success(
                labels2
            )
    #keterangan = np.array(0)
    # st.write(prediksi[keterangan])

    # st.subheader(
    #   'Probabilitas Hasil Prediksi (Klasifikasi Penerima Bantuan PKH)')
    # st.write(pred_prob)


run()
