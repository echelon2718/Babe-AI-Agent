from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

import google.generativeai as genai
import re
import json
import pandas as pd
import ast

genai.configure(api_key="YOUR_GEMINI_API_KEY")

reconfirm_translator_prompt = '''
Kamu adalah AI Agent untuk Kulkas Babe (@kulkasbabe.id), sebuah brand retail alkohol yang fokus pada delivery order.

Tugasmu adalah sebagai berikut:
1. Menerjemahkan pesan yang memiliki format seperti:

RECONFIRM JAJAN
Nama: Arendra (Harus ada)
Nomor Telepon: 08123456789 (Harus ada)
Produk : Atlas Lychee 2 Botol Promo (Item/Paket, Harus ada, boleh lebih dari satu produk)
Alamat : Jl. Ahmad Saleh No. 123, Jakarta (Harus ada)
Payment : Transfer (Harus ada, pilihannya Transfer, cash, atau split bill)
Tuker Voucher : Tumblr (Tidak wajib, jika tidak ada, kosongkan saja)
Notes : Es Batu, Sticker 3 (Tidak wajib, jika tidak ada, kosongkan saja)

Menjadi JSON dengan format dan ketentuan sebagai berikut:

{
    "cust_name" : <str, dynamic>,
    "phone_num" : <str, dynamic>,
    "ordered_products: <dict<str, dynamic>>,
    "address" : <str, dynamic>,
    "payment_type" : <str, categorical>,
    "tuker_voucher" :<dict<str, dynamic>>,
    "notes": <str, dynamic>
}

1. cust_name: nama kastamer
2. phone_num: nomor telepon kastamer

3. ordered_products: Dict yang berisi produk-produk. Isinya akan berbentuk seperti ini:
{
        {
            "tipe": <str>,
            "produk": <str>,
            "quantity": <int>,
        },
}

Contoh:

{
        {
            "tipe" : "Paket",
            "produk" : "Atlas Lychee 2 Botol Promo",
            "quantity": 1,
        },

        {
            "tipe" : "Item",
            "produk" : "Singleton 500 mL",
            "quantity" : 3,
        }
}

4. address: Berisi alamat kastamer untuk menentukan jarak
5. payment_type: Jenis pembayaran
6. tuker_voucher: Berisi item yang mau dituker dengan voucher. Bentuknya seperti ini:
{
        {
            "item": <str, dynamic>,
            "quantity": <int, dynamic>
        }
}
    Contoh:
    {
        {
                "item": "Tumblr",
                "quantity": 1
        },
        {
                "item": "Stiker",
                "quantity": 3
        }
    }

7. notes: Berisi catatan dari klien, biasanya ini bisa meminta suatu produk tambahan seperti:
es batu, stiker. Kemungkinan maksimumnya, klien mungkin akan memesan es batu, atau stiker. Jika iya,
susun notes seperti ini:
{
    {
        "item": "Es Batu",
        "quantity": 1 (misal)
    },
    {
        "item": "Stiker",
        "quantity": 3 (misal)
    }
}

Jika klien memiliki request khusus seperti meminta belikan rokok, atau makanan diluar toko, atau apapun yang meminta driver untuk melakukan perjalanan diluar pengantaran,
maka berikan notes sebagai berikut:
{
    "notes": "Permintaan khusus: {isi pesan dari pelanggan}",
    "nitip": True
}

Namun jika ada permintaan yang tidak sesuai dengan ketentuan ini, maka akan
dikembalikan dengan value: "Fallback ke manusia: {isi pesan dari pelanggan}"

KALAU FORMAT RECONFIRM JAJAN KURANG SESUAI (Beri sedikit toleransi jika artinya sama) ATAU ADA DATA YANG TIDAK LENGKAP, KEMBALIKAN DENGAN FORMAT JSON SEBAGAI BERIKUT:
{
    "fallback": "Ada data yang kurang sesuai, pastikan format pesan sesuai ini:
    RECONFIRM JAJAN
    Nama: [Harus ada]
    Nomor Telepon: [Harus ada]
    Jenis Pesanan: [Harus ada, pilihannya Paket atau Item]
    Produk : [Harus ada, boleh lebih dari satu produk]
    Alamat : [Harus ada]
    Payment : [Harus ada, pilihannya Transfer, cash, atau split bill]
    Tuker Voucher : [Tidak wajib, jika tidak ada, kosongkan saja]
    Notes : [Tidak wajib, jika tidak ada, kosongkan saja]

    "
}

ANDA HANYA BOLEH MENJAWAB DENGAN FORMAT JSON SEBAGAIMANA DIATAS, JIKA ADA PESAN YANG TIDAK SESUAI ATAU TIDAK JELAS,
JAWAB DENGAN "Fallback ke manusia: {isi pesan dari pelanggan}".
'''

data_retriever_prompt = '''
Anda adalah sebuah agent yang bertugas untuk memilih indeks data yang paling sesuai dengan prompt yang diberikan.
Cukup jawab dengan nomor indeks data yang paling sesuai dengan prompt yang diberikan.
Tidak perlu menjelaskan apapun, cukup jawab dengan nomor indeks data yang paling sesuai dengan format yang telah ditentukan.
Jika ada produk yang punya tulisan "- I" atau "- O", utamakan "- I" dahulu.
'''

class AgentRetriever:
    def __init__(
            self, 
            gemini_api_key="AIzaSyCqytJHKzR-nWZPJNXG0bZ6SMh_3DFtGJE", 
            product_metadata=None,
            combo_metadata=None,
            product_db=None,
            combo_db=None,
        ):
        self.gemini_api_key = gemini_api_key
        self.product_metadata = product_metadata
        self.combo_metadata = combo_metadata
        self.product_db = product_db
        self.combo_db = combo_db
        self.product_df = pd.read_csv(self.product_db) if self.product_db else None
        self.combo_df = pd.read_csv(self.combo_db) if self.combo_db else None
        self.product_metadata_df = pd.read_csv(self.product_metadata) if self.product_metadata else None
        self.combo_metadata_df = pd.read_csv(self.combo_metadata) if self.combo_metadata else None

        genai.configure(api_key=gemini_api_key)
    
    def clean_llm_json_output(self, text: str) -> dict:
        # Hilangkan ```json ... ```
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        
        # Parsing ke dict
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("[ERROR] Gagal parse JSON:", e)
            return None

    def retrieve(self, query, df, top_k=5, id_col='id', search_column='name'):
        tokenized_corpus = [word_tokenize(prod.lower()) for prod in df[search_column]]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)

        df['bm25_score'] = scores
        hasil = df.sort_values(by='bm25_score', ascending=False)[:top_k]
        hasil_clean = hasil[[id_col, search_column]].to_dict(orient='records')
        # print(f"Top {top_k} results: {json.dumps(hasil_clean, indent=2)}")

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=data_retriever_prompt,
        )

        gemini_ans = model.generate_content(f"{hasil_clean}")
        try:
            gemini_ans = int(gemini_ans.text)
            return gemini_ans
        except:
            print("[ERROR] Gagal mendapatkan indeks dari Gemini:", gemini_ans.text)
            return None
        
    def reconfirm_translator(self, message):
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=reconfirm_translator_prompt,
        )

        gemini_ans = model.generate_content(message)
        sanitized_response = self.clean_llm_json_output(gemini_ans.text)

        return sanitized_response

    def run(self, query):
        reconfirm_json = self.reconfirm_translator(query)
        print("PESANAN RECONFIRM")
        print("Nama Pelanggan:", reconfirm_json['cust_name'])
        print("Nomor Telepon:", reconfirm_json['phone_num'])
        print("Alamat:", reconfirm_json['address'])
        print("-" * 60)
        print(f"{'JENIS':<5} {'BARANG':<40} {'ID':<10}")
        print("-" * 60)
        for item in reconfirm_json['ordered_products']:
            # Cek apakah item adalah paket atau item
            if item['tipe'] == 'Paket':
                df = self.combo_metadata_df
                idx = self.retrieve(item['produk'], df, id_col='id', search_column='name')
                # indice = self.combo_df[self.combo_df['id'] == idx].reset_index(drop=True) # Mungkin saja lebih dari satu produk karena terdiri atas beberapa varian
                jawaban = f"PAKET {self.combo_df[self.combo_df['id'] == idx]['name'].iloc[0]:<40} {idx:<10}"
            else:
                df = self.product_df
                idx = self.retrieve(item['produk'], df, id_col='id', search_column='name')
                indice = df[df['id'] == idx].reset_index(drop=True)

                # Cek apakah ada variant
                if indice['variants'][0] != '[]':
                    variants = ast.literal_eval(indice['variants'][0])
                    variants = pd.DataFrame(variants)
                    variants_ready = variants[variants['stock_qty'] > 0]
                    prioritas = ['L', 'P', 'C']
                    jawaban = ""

                    for prefix in prioritas:
                        hasil = variants_ready[variants_ready['name'].str.startswith(prefix)]
                        if not hasil.empty:
                            product_name = self.product_df[self.product_df['id'] == hasil['product_id'].iloc[0]]['name'].iloc[0]
                            product_id = hasil['product_id'].iloc[0]
                            variant_id = hasil['id'].iloc[0]
                            jawaban = f"ITEM  {product_name:<40} {product_id}|{variant_id:<10}"

                else:
                    jawaban = f"ITEM {indice['name'][0]:<40} {indice['id'][0]:<10}"


            print(jawaban)
