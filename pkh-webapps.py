import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.write("""
# Sistem Rekomendasi Ekspor Untuk UMKM""")

img = Image.open('ekspor.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

knnmodel = pickle.load(open('./model/model_fcm2.pkl', 'rb'))


def run():
    jenis = {25: 'Biji kakao',
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
    jn = list(jenis.keys())
    jenis_barang = st.selectbox(
        'Jenis Barang', jn, format_func=lambda x: jenis[x])

    negara = st.text_input('Input Negara Tujuan Ekspor')

    skala = {5: 'skala besar mencapai puluhan juta unit',
             4: 'skala besar mencapai ratusan ribu unit',
             3: 'skala sedang mencapai puluhan ribu unit',
             2: 'skala kecil mencapai ribuan unit',
             1: 'skala rend ah mencapai ratusan unit'}
    sk = list(skala.keys())
    skala_pengiriman = st.selectbox(
        'Skala Pengiriman Barang', sk, format_func=lambda x: skala[x])

    harga = st.number_input(
        'Input Harga Barang Yang Akan Dijual (Satuan Dollar)')
    tax = st.number_input(
        'Input Nilai Pajak Maksimal Negara Tujuan yang Diinginkan')

    # # code prediksi
    # rekom_diagnosis = ''

    # # membuat tombol untuk prediksi
    # if st.button('Rekomendasi Negara Tujuan Ekspor'):
    #     rekom_prediction = knnmodel.predict([[jenis, skala, harga, tax]])

    st.subheader('Tabel Inputan Data')

    data = {'Skala': skala_pengiriman,
            'Harga Barang': harga,
            'Pajak': tax}
    fitur = pd.DataFrame(data, index=[0])
    st.write(fitur)

    #prediksi = modelnb.predict(fitur)
    pred_prob = knnmodel.predict_proba(fitur)
    keterangan = np.array(
        ['Sangat Direkomendasikan', 'Tidak Direkomendasikan'])

   # st.subheader('Keterangan Label Kelas')

   # st.write(keterangan)

    st.subheader('Hasil Rekomendasi Negara Tujuan Ekspor')
    if st.button("Submit"):
        fitur = pd.DataFrame(data, index=[0])
        print(fitur)
        prediction = knnmodel.predict(fitur)
        keterangan = np.array(['LAYAK', 'TIDAK LAYAK'])
        lc = [str(i) for i in prediction]
        keterangan = np.array(0)
        labels2 = (prediction[keterangan])
        if keterangan == 0:
            st.error(
                'Tidak Direkomendasikan'
            )
        else:
            st.success(
                'Sangat Direkomendasikan'
            )


run()
