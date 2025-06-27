from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import google.generativeai as genai
import re
import json
import pandas as pd
import requests
import ast
from modules.crud_utility import add_prod_to_order, update_payment, fetch_product_combo_details, fetch_order_details, update_order_detail, cek_kastamer, create_order, list_payment_modes, update_status, cetak_struk, fetch_product_item_details, update_order_attr, void_order
from modules.maps_utility import resolve_maps_shortlink, get_travel_distance, address_to_latlng, distance_cost_rule, is_free_delivery, estimasi_tiba
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any

nltk.download('punkt')
nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

payment_dict = {
    "Cash": 0,
    "BRI": 1,
    "Hutang": 2,
    "BCA": 3,
    "QRIS": 4,
}

reconfirm_translator_prompt = '''
Kamu adalah AI Agent untuk Kulkas Babe (@kulkasbabe.id), sebuah brand retail alkohol yang fokus pada delivery order.

Tugasmu adalah sebagai berikut:
1. Menerjemahkan pesan yang memiliki format seperti:

RECONFIRM JAJAN
Nama: Arendra (Harus ada)
Nomor Telepon: 08123456789 (Harus ada)
Produk: Atlas Lychee 2 Botol Promo (Item/Paket, Harus ada, bisa lebih dari satu produk)
Alamat: Jl. Ahmad Saleh No. 123, Jakarta (Harus ada)
Payment: BCA (Harus ada, pilihannya BCA, BRI, Cash, QRIS, atau Hutang. Jika selain ini, anggap sebagai Cash)
Tukar Voucher: Tumblr (Tidak wajib, jika tidak ada, kosongkan saja; default quantity = 1)
Notes: Es Batu, Sticker 3 (Tidak wajib, jika tidak ada, kosongkan saja)
Disc: 10% (Tidak wajib, jika tidak ada, kosongkan saja)
Pengiriman: FD (Tidak wajib, default "FD", jika tidak disebutkan, pilihannya FD, I, EX)

Jika ada permintaan untuk melakukan voiding seperti, "batalkan struk <ID>", "void struk <ID>", atau "batalkan order <ID>", maka kembalikan pesan dengan format:
{
    "pembatalan": "<list ID>"
}

Tapi pastikan ada IDnya, jika tidak ada ID, kembalikan pesan dengan format:
{
    "fallback": "Tidak ada ID yang diberikan untuk pembatalan."
}

2. Output harus berupa JSON valid, dengan struktur dan aturan berikut:

{
  "cust_name": <str>,
  "phone_num": <str>, (normalisasikan nomor telepon, hapus spasi, strip, atau karakter lain yang tidak perlu, dengan format 08XXXXXX)
  "mode_diskon": <"number"|"percentage"> (default "percentage", jika tidak ada mode diskon, isi dengan "percentage". WAJIB!!),
  "disc": <float> (default 0.0, jika tidak ada disc, kosongkan saja),
  "ordered_products": [
    {
      "tipe": <"Paket"|"Item">,
      "produk": <str>,
      "quantity": <int>
    },
    ...
  ],
  "address": <str>,
  "payment_type": <"BCA"|"BRI"|"Cash"|"QRIS"|"Hutang">,
  "notes": <str>
  "jenis_pengiriman":<"FD"|"EX"|"I"> (default "FD")
  "status": <"Lunas"|"Pending"> (default "Lunas")
}

Keterangan:
- cust_name: Nama pelanggan.
- phone_num: Nomor telepon pelanggan.
- mode_diskon: "number" jika diskon dalam format angka (misal 1000 atau 1K atau 1k), "percentage" jika diskon dalam format persentase (misal 10% = 0.1). Jika tidak ada diskon, defaultnya "percentage".
- disc: Diskon dalam format desimal (misal 10% = 0.1). Jika tidak ada diskon, kosongkan saja.
- ordered_products: Array objek, satu per produk atau voucher:
  a.) tipe: "Paket" untuk produk berjenis paket/voucher (lihat daftar di bawah), "Item" untuk produk perbiji atau tambahan seperti es batu atau stiker. INGAT!! bahwa "Item" tidak mungkin berisi lebih dari satu produk dalam satu kuantitas. Jadi jika user meminta "Atlas Lychee 2 Botol Promo (Paket)", maka tipe-nya adalah "Paket". Jika user meminta "3 Anggur Merah 500 mL (Item)", maka tipe-nya adalah "Item".
  b.) produk: Nama paket atau item persis seperti di pesan.
  c.) quantity: Jumlah. Jika pelanggan tidak mencantumkan angka, default 1.

- address: Alamat untuk pengiriman.
- payment_type: Harus salah satu dari BCA, BRI, Cash, QRIS, Hutang. Lainnya => Cash.
- notes: Semua permintaan khusus selain ordered_products. Jika pelanggan meminta rokok atau bawain sesuatu, atau layanan di luar delivery alkohol, jadikan notes serta tambahkan item "Nitip ke Jagoane Babe (Item)" dengan quantity 1, kecuali jika minta es batu, cup, atau stiker, maka tambahkan sebagai produk terpisah dengan tipe "Item" dan quantity sesuai permintaan.
- PASTIKAN UNTUK MEMETAKAN INPUT KE TIPE "Item" JIKA ADA KURUNG BERTULISAN (Item) ATAU KE TIPE "Paket" JIKA ADA KURUNG BERTULISAN (Paket). Misal: "3 botol Atlas Lychee 600 mL (Item)" => tipe "Item", quantity 3, "4 paket Atlas Lychee 2 Botol Promo (Paket)" => tipe "Paket", quantity 4.

- jenis_pengiriman: Jenis pengiriman, FD: free delivery, I: instant delivery, EX: express. Nilai defaultnya "FD", jika tidak disebutkan. Jika FD, tambahkan Garansi ke dalam ordered_products. Jika I, tambahkan "Instant Delivery (Paket)" ke dalam ordered_products quantity 1, tanpa Garansi. Jika EX, tambahkan "Express Delivery!! (Paket)" ke dalam ordered_products quantity 1, tanpa Garansi. 
- status: Status pembayaran, "Lunas" jika sudah bayar, "Pending" jika belum bayar. Nilai defaultnya "Lunas", jika tidak disebutkan.
CATATAN PENTING: Garansi hanya boleh ada 1 kali dalam ordered_products, quantitynya juga harus 1 saja. Ini berlaku baik "Garansi" maupun "Babe Garansi-in!!!".

3. Daftar tipe "Paket" tambahan:
- Merch / Merh Babe / Polos XXX (XXX ini angka, misal "Merch Babe 1", "Merh Polos 2". Jadi jika pelanggan meminta "Merch Babe 2" (atau "Merch 2" ini artinya sama dengan "Merch Babe 2"), maka tipe-nya adalah "Paket" dan produk-nya adalah "Merch Babe 2", quantity = 1, bukan "Merch Babe", quantity = 2)
- Babe Garansi-in!!! / Garansi
- Tukar Kupon / Voucher
- Komplimen XXX
- Delivery
- Hadiah XXX

4. Daftar tipe "Item" tambahan:
- Es Batu (Ini dianggap sebagai item yang harus dimasukkan ke dalam order meskipun ditaruh di notes, jadi jika pelanggan meminta "Es Batu 3", maka tipe-nya adalah "Item", produk-nya adalah "Es Batu", quantity = 3)
- Cup (Sama seperti es batu, jika pelanggan meminta "Cup 2", maka tipe-nya adalah "Item", produk-nya adalah "Cup Babe", quantity = 2. HARUS DITULIS Cup Babe)
- Stiker (Jika user minta ini, dimasukkan saja sebagai notes beserta jumlahnya pakai bahasa natural, tidak perlu dimasukan ke dalam products)
- Nitip ke Jagoane Babe (Item) (Jika pelanggan meminta rokok, menitip atau minta bawakan sesuatu, atau layanan di luar delivery alkohol, tambahkan produk ini dengan quantity 1, kecuali jika minta es batu, cup, atau stiker, maka tambahkan sebagai produk terpisah dengan tipe "Item" dan quantity sesuai permintaan)

5. Fallback: Jika format RECONFIRM JAJAN tidak sesuai atau ada data wajib yang hilang, kembalikan JSON berikut saja:
{
  "fallback": "Ada data yang kurang atau format tidak sesuai. Pastikan format pesan:

RECONFIRM JAJAN
Nama: Arendra (Harus ada)
Nomor Telepon: 08123456789 (Harus ada)
Produk: Atlas Lychee 2 Botol Promo (Item/Paket, Harus ada, bisa lebih dari satu produk)
Alamat: Jl. Ahmad Saleh No. 123, Jakarta (Harus ada)
Payment: BCA (Harus ada, pilihannya BCA, BRI, Cash, QRIS, atau Hutang. Jika selain ini, anggap sebagai Cash)
Tukar Voucher: Tumblr (Tidak wajib, jika tidak ada, kosongkan saja)
Notes: Es Batu, Sticker 3 (Tidak wajib, jika tidak ada, kosongkan saja)
Disc: 10% (Tidak wajib, jika tidak ada, kosongkan saja)
Pengiriman: FD (Tidak wajib, default "FD", jika tidak disebutkan, pilihannya FD, I, EX)
"

6. Contoh:

Pesan Pelanggan:

RECONFIRM JAJAN
Nama: Budi
Nomor Telepon: 08987654321
Produk: Atlas Lychee 2 Botol Promo, Singleton 500 mL (Item) 3, 3 botol Anggur Merah 500 mL (Item)
Tuker Voucher: Tumblr
Alamat: Jl. Melati No. 5, Bandung
Payment: QRIS
Disc: 10%
Notes: Es Batu, Sticker 2, nitip rokok

EX, lunas

Contoh Output JSON BENAR:
{
  "cust_name": "Budi",
  "phone_num": "08987654321",
  "disc": 0.1,
  "ordered_products": [
    {
      "tipe": "Paket",
      "produk": "Atlas Lychee 2 Botol Promo",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Singleton 500 mL",
      "quantity": 3
    },
    {
      "tipe": "Item",
      "produk": "Anggur Merah 500 mL",
      "quantity": 3
    },
    {
      "tipe": "Paket",
      "produk": "Tuker Voucher: Tumblr",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Es Batu",
      "quantity": 1
    },
    {
      "tipe": "Paket",
      "produk": "Express Delivery!!",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Nitip ke Jagoane Babe",
      "quantity": 1
    }
  ],
  "address": "Jl. Melati No. 5, Bandung",
  "payment_type": "QRIS",
  "notes": "Tambahan stiker 2, nitip rokok",
  "jenis_pengiriman": "EX",
  "status": "Lunas"
}

Contoh Output JSON SALAH (fallback):
{
  "cust_name": "Budi",
  "phone_num": "08987654321",
  "disc": 0.1,
  "ordered_products": [
    {
      "tipe": "Paket",
      "produk": "Atlas Lychee 2 Botol Promo",
      "quantity": 2
    },
    {
      "tipe": "Item",
      "produk": "3 Singleton 500 mL",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "3 botol Anggur Merah 500 mL",
      "quantity": 1
    },
    {
      "tipe": "Paket",
      "produk": "Tuker Voucher: Tumblr",
      "quantity": 1
    },
    {
      "tipe": "Item",
      "produk": "Es Batu",
      "quantity": 1
    },
  ],
  "address": "Jl. Melati No. 5, Bandung",
  "payment_type": "QRIS",
  "notes": "Tambahan stiker 2, nitip rokok",
  "jenis_pengiriman": "EX",
  "status": "Lunas"
}

CATATAN DISKON TAMBAHAN:
- Diskon Atensi: 100%
- Diskon Giveaway: 100%
- Diskon Komplimen: 100%
- Diskon KOL: 100%
- Diskon Media Partner: 100%
- Diskon Ngacara: 100%
- Diskon RND: 100%
'''

item_selection_prompt = '''
Anda adalah sebuah agent yang bertugas untuk memilih indeks data yang paling sesuai dengan prompt yang diberikan.
Cukup jawab dengan nomor indeks data yang paling sesuai dengan prompt yang diberikan.
Tidak perlu menjelaskan apapun, cukup jawab dengan nomor indeks data yang paling sesuai dengan format yang telah ditentukan.
Jika ada produk yang punya tulisan "- I" atau "- O", utamakan "- I" dahulu. Jangan pilih "- O" jika tidak dituliskan secara eksplisit.

CATATAN: JIKA TIDAK ADA PRODUK YANG SESUAI, JAWAB -99999
'''

combo_selection_prompt = '''
Anda adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memetakan permintaan pelanggan ke indeks paket yang tepat dalam database.

Aturan Utama
1. Anda akan diberikan List berisi JSON yang berisi daftar paket. Misal:
[{'id': 256319, 'name': 'Paket 2 Atlas Lychee [Promo April]'}, {'id': 256487, 'name': 'Paket 3 Botol Anker Lychee [Promo April]'}, {'id': 256548, 'name': 'Paket 3 Botol Vibe Lychee 700ml [Promo April]'}]
2. Cari satu entri dalam daftar tersebut yang paling cocok dengan query secara semantik:
- Pertimbangkan kuantitas: angka dapat berada setelah kata "Paket" (misal "Paket 2 ...") atau sebelum kata "Botol" (misal "2 Botol ...").
- Nama produk: cocokkan merek dan varian persis seperti input user (abaikan kapitalisasi).
- Promo dinamis: setiap paket diakhiri tag [Promo X], di mana X bisa nama bulan atau event. Pilih promo yang sesuai dengan kata setelah "Promo" dalam input user.
- Kombinasi multi-produk dipisah dengan tanda "+"; hanya cocokkan jika input user menyebutkan kedua komponen serta kuantitas masing-masing.
3. Jawab hanya dengan nomor indeks entri yang paling sesuai. Tidak perlu penjelasan atau teks tambahan.
4. Jika tidak ada entri yang sesuai, jangan beri jawaban apapun.

Contoh Daftar Paket (Ini jika tidak dalam bentuk JSON, misal komponen kiri merepresentasikan id nya dan diikuti nama paketnya):
13841. Paket 2 Daebak Soju Lemon [Promo Juni]
13252. Paket 2 Smirnoff Ice Lemon [Promo April]
35122. 6 Botol Atlas Rose Pink [Promo Juni]
45231. 10 Botol Cheosnun Green Grape [Promo Juni]
43576. 12 Singaraja Beer 620ml [Promo Juni]
88431. Paket Kolesom Biasa + Draft Beer [Promo Juni]
34963. Paket Anggur Merah Gold + Draft Beer [Promo Juni]
54133. Paket 2 Kawa Merah Gold + 2 Draft Beer [Promo Juni]
13412. Paket 3 Anggur Merah Biasa [Promo Juni]
55232. Paket QRO + Bintang Pilsener 620ml [Promo Juni]
33135. Paket Atlas Rose Pink + Singaraja 620ml [Promo Juni]
42551. Sababay Pink Blossom [Promo Juni]
23453. Iceland Vodka Lychee 500ml [Promo Juni]

Contoh Interaksi
Input: "Smirnoff Ice Lemon 2 Botol Promo April"
Output: 13252 (karena sesuai format "Paket 2 Smirnoff Ice Lemon [Promo April]")

Input: "6 Botol Atlas Rose Pink Promo Juni"
Output: 35122

Input: "Kawa Merah Gold 2 Draft Beer Promo Juni"
Output: 54133

JIKA TIDAK ADA PAKET YANG SESUAI, JAWAB -99999
PEMILIHAN ITEM TIDAK BOLEH SALAH TERUTAMA DALAM HAL KUANTITAS!!!!
'''

merch_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri merch dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri garansi yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA MERCH YANG SESUAI, JAWAB -99999
'''

garansi_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri garansi dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri garansi yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA GARANSI YANG SESUAI, JAWAB -99999
'''

kupon_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri kupon dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri kupon yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA KUPOON YANG SESUAI, JAWAB -99999
'''

voucher_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri voucher dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri voucher yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA VOUCHER YANG SESUAI, JAWAB -99999
'''

komplimen_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri komplimen dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri voucher yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA KOMPLIMEN YANG SESUAI, JAWAB -99999
'''

delivery_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri delivery dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri delivery yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA DELIVERY YANG SESUAI, JAWAB -99999
'''

hadiah_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri hadiah dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri hadiah yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
JIKA TIDAK ADA HADIAH YANG SESUAI, JAWAB -99999
'''

notes_prompt = '''
Beri salam hangat kepada pelanggan, tidak usah menggunakan bold atau bahasa terlalu formal.
Sebutkan paket-paket promo yang dipesan. Jawab dengan gaya persis seperti di bawah ini.

Contoh input:
RECONFIRM JAJAN
Nama: Kevin
Nomor Telepon: 085853605806
Produk: 3 botol QRO anggur merah 650 ml (Item); Paket 3 AO Mild (Paket); 1 buah Paket 2 Anggur Hijau MCD + Kawa (Paket)
Alamat: Jl. Melati No. 5, Bandung
Disc: 10%
Payment: BCA
Notes: Es Batu 3

Contoh output:

"Makasih yaa (nama) niii Jajan mu langsung ta proses duluu. Paket nyaa (sebutkan paketnya) yah."
'''

task_instructions = {
    "reconfirm_translator_prompt": reconfirm_translator_prompt,
    "item_selection_prompt": item_selection_prompt,
    "combo_selection_prompt": combo_selection_prompt,
    "merch_selection_prompt": merch_selection_prompt,
    "garansi_selection_prompt": garansi_selection_prompt,
    "kupon_selection_prompt": kupon_selection_prompt,
    "voucher_selection_prompt": voucher_selection_prompt,
    "komplimen_selection_prompt": komplimen_selection_prompt,
    "delivery_selection_prompt": delivery_selection_prompt,
    "hadiah_selection_prompt": hadiah_selection_prompt,
    "notes_prompt": notes_prompt,
}

class AgentBabe:
    def __init__(
      self,
      instructions: dict = task_instructions,
      model_name: str = "gemini-1.5-flash",
      df_product_dir: str = "./kulkasbabe.csv",
      df_combo_dir: str = "./paket.csv",
      top_k_retrieve: int = 5,
      gmap_api_key: Optional[str] = None,
    ):
        self.instructions = instructions
        self.model_name = {
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro",
        }
        self.df_product_dir = df_product_dir
        self.df_combo_dir = df_combo_dir
        self.top_k_retrieve = top_k_retrieve
        self.longlat_toko = (-7.560745951139057, 110.8493297202405)
        self.gmap_api_key = gmap_api_key
        self.free_areas = ["Gedongan", "Gedangan", "Gentan", "Kadilangu", "Kudu", "Kwarasan",
                           "Langenharjo", "Madegondo", "Gonilan", "Gumpang", "Pabelan", "Blulukan",
                           "Karangasem", "Baturan", "Gajahan", "Paulan"]

        self.product_df = pd.read_csv(self.df_product_dir)
        self.combo_df = pd.read_csv(self.df_combo_dir)
    
    def clean_llm_json_output(self, text: str) -> dict:
        # Hilangkan ```json ... ```
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        
        # Parsing ke dict
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("[ERROR] Gagal parse JSON:", e)
            return None
    
    def select_id_by_agent(
        self,
        query: str,
        df: pd.DataFrame,
        task_instruction: str,
        id_col: str = 'id',
        evaluation_col: str = 'name',
    ):
        tokenized_corpus = [word_tokenize(prod.lower()) for prod in df[evaluation_col]]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(query.lower())
        sim_score = bm25.get_scores(tokenized_query)
        df['bm25_score'] = sim_score
        
        sim_score_table = df.sort_values(by='bm25_score', ascending=False)[:self.top_k_retrieve]
        sim_score_table = sim_score_table[[id_col, evaluation_col]].to_dict(orient='records')

        ######## DEBUG ########
        # print(f"[DEBUG] Query untuk BM25: {query}")
        # print(f"[DEBUG] Top-{self.top_k_retrieve} kandidat hasil BM25:")
        # print(sim_score_table)
        #######################

        LLM = genai.GenerativeModel(
            model_name=self.model_name['pro'],
            system_instruction=task_instruction,
        )
        idx = LLM.generate_content(f"Query: {query}, List: {sim_score_table}")
        # print("[DEBUG] Hasil dari LLM untuk ID:", idx.text)

        try:
            idx = int(idx.text)
            # print("[DEBUG] ID terpilih:", idx)
            return idx
        except ValueError:
            print("[ERROR] Gagal mendapatkan indeks dari Gemini:", idx.text)
            return None
    
    def reconfirm_translator(self, message):
        model = genai.GenerativeModel(
            model_name=self.model_name['flash'],
            system_instruction=self.instructions['reconfirm_translator_prompt'],
        )

        gemini_ans = model.generate_content(message)
        sanitized_response = self.clean_llm_json_output(gemini_ans.text)

        return sanitized_response

    def _process_item(
        self,
        order_id: str,
        nama_produk: str,
        qty: int,
        access_token: str,
    ):
        df = self.product_df[self.product_df['pos_hidden'] == 0]
        idx = self.select_id_by_agent(nama_produk, df, self.instructions['item_selection_prompt'], id_col='id', evaluation_col='name')
        if idx is None or idx == -99999:
            logger.error("Gagal menemukan produk item: %s", nama_produk)
            update_status(order_id, "X", access_token=access_token)
            return False, f"Gagal menemukan produk item {nama_produk}, tolong masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan"

        df_sel = df[df['id'] == idx].reset_index(drop=True)

        if df_sel.at[0, 'variants'] != '[]':
            try:
                variants = ast.literal_eval(df_sel.at[0, 'variants'])
                variants_df = pd.DataFrame(variants)
            except Exception as e:
                logger.error("Gagal parse variants untuk produk %s: %s", idx, e)
                return False, "Gagal memproses varian produk."
            variants_ready = variants_df[variants_df['stock_qty'] > 0]
            for prefix in ['P', 'L', 'C']:
                sel = variants_ready[variants_ready['name'].str.startswith(prefix)]
                if not sel.empty:
                    variant_id = sel['id'].iloc[0]
                    idx = f"{idx}|{variant_id}"
                    logger.debug("Varian dipilih untuk item %s: %s", nama_produk, idx)
                    break
        
        # Tambah ke order
        try:
            resp = add_prod_to_order(order_id, idx, qty, access_token=access_token)
            # Cek apakah ada key error
            if resp.get('error'):
                logger.error("Error saat menambahkan item ke order: %s", resp['error']['message'])
                update_status(order_id, "X", access_token=access_token)
                return False, resp['error']['message']
            # Kalau perlu cek resp.status_code atau resp.json untuk deteksi error
            logger.debug("Response add_prod_to_order: %s", getattr(resp, 'text', ''))
            print(f"ITEM  {df_sel['name'].iloc[0]:<45} {idx:<15} {qty:<15}")
            return True, "Item berhasil ditambahkan ke order."
        except requests.exceptions.HTTPError as http_err:
            logger.error("HTTPError add item: %s", http_err)
            update_status(order_id, "X", access_token=access_token)
            return False, "Ada kesalahan HTTP saat memasukkan item ke order. struk di-voidkan."
        except Exception as e:
            logger.error("Error lain add item: %s", e)
            return False, f"Ada kesalahan saat memasukkan item ke order. Error: {e}"
    
    def _process_paket(self, order_id: Any, nama_paket: str, qty_pesan: int, access_token: str) -> Any:
        """
        Proses paket: cari ID paket, fetch detail, tambahkan tiap item dalam paket,
        lalu update diskon tiap item. 
        Mengembalikan True jika sukses, atau string pesan error jika gagal.
        """
        # Filter combo yang tidak hidden
        df = self.combo_df[self.combo_df['pos_hidden'] == 0]
        task_instruction = self.instructions['combo_selection_prompt']
        lower = nama_paket.strip().lower()

        # Sesuaikan filter berdasarkan pola nama
        if lower.startswith(('merch', 'mer')):
            mask = df['name'].str.lower().str.contains('merch|merh')
            df = df[mask]
            task_instruction = self.instructions['merch_selection_prompt']
        elif lower.startswith(('babe garansiin','garansi','garan')):
            df = df[df['name'] == 'Babe Garansi-in !!!']
            task_instruction = self.instructions['garansi_selection_prompt']
        elif 'kupon' in lower:
            df = df[df['name'].str.lower().str.contains('kupon')]
            task_instruction = self.instructions['kupon_selection_prompt']
        elif 'voucher' in lower:
            df = df[df['name'].str.lower().str.contains('voucher')]
            task_instruction = self.instructions['voucher_selection_prompt']
        elif lower.startswith(('komplimen','komp')):
            df = df[df['name'].str.lower().str.startswith(('komplimen','komp'))]
            task_instruction = self.instructions['komplimen_selection_prompt']
        elif 'delivery' in lower:
            df = df[df['name'].str.lower().str.contains('delivery')]
            task_instruction = self.instructions['delivery_selection_prompt']
        elif lower.startswith('hadiah'):
            df = df[df['name'].str.lower().str.startswith('hadiah')]
            task_instruction = self.instructions['hadiah_selection_prompt']

        if df.empty:
            logger.error("Tidak ada paket matching untuk: %s", nama_paket)
            return f"Gagal menemukan paket: {nama_paket}"
        if len(df) == 1:
            paket_id = df['id'].iloc[0]
        else:
            paket_id = self.select_id_by_agent(nama_paket, df, task_instruction, id_col='id', evaluation_col='name')
        if paket_id is None:
            logger.error("select_id_by_agent gagal untuk paket: %s", nama_paket)
            return f"Gagal menemukan paket: {nama_paket}"

        if paket_id == -99999:
            logger.error("Tidak ada paket yang sesuai dengan nama: %s", nama_paket)
            update_status(order_id, "X", access_token=access_token)
            return f"Gagal menemukan paket yang sesuai: {nama_paket}, tolong masukkan dengan format <Nama produk> (<QTY> Paket/Item), dan hindari penggunaan singkatan (AI tidak tahu konteks dalam singkatan itu). Sebisa mungkin, sertakan juga brand-nya apa agar menghindari kesalahpahaman AI, misal AM bisa dianggap dari Mix Max Anggur Merah, QRO Anggur Merah, atau Kawa Kawa Anggur Merah, tapi kalau ini tidak dianggap masalah, silakan diabaikan. CONTOH: 2 Atlas Lychee + 2 Beer (1 Paket). Jika error ini masih berlangsung, cek backoffice Olsera. Struk di-voidkan."

        # Ambil detail paket
        try:
            combo_details = fetch_product_combo_details(paket_id, access_token)
            combo_items = combo_details['data']['items']['data']
        except Exception as e:
            logger.error("Gagal fetch combo details untuk %s: %s", paket_id, e)
            return "Gagal ambil detail paket."

        total_harga_normal = 0.0
        # Tambah tiap item
        for item in combo_items:
            product_id = item.get('product_id')
            var_id = item.get('product_variant_id')
            qty_total = item.get('qty', 0) * qty_pesan
            # Fetch detail item untuk cek stok & harga
            try:
                item_details = fetch_product_item_details(product_id, access_token=access_token)
                data = item_details.get('data', {})
            except Exception as e:
                logger.error("Gagal fetch item_details ID %s: %s", product_id, e)
                return f"Gagal ambil data produk {product_id}."
            # Cek variant dan harga
            if not data.get('variant'):
                idx = f"{product_id}"
                harga = float(data.get('sell_price_pos', 0))
            else:
                variants = data.get('variant', [])
                item_df = pd.DataFrame(variants)
                harga = float(item_df['sell_price_pos'].iloc[0]) if not item_df.empty else 0.0
                if 'stock_qty' not in item_df.columns:
                    logger.error("Data stok tidak ada untuk produk %s", product_id)
                    return f"Data stok tidak ditemukan untuk {item.get('product_name')}."
                variants_ready = item_df[item_df['stock_qty'] >= qty_total]
                if variants_ready.empty:
                    logger.error("Stok tidak cukup untuk produk %s", product_id)
                    return f"Maaf, stok tidak cukup untuk {item.get('product_name')}."
                # Pilih varian: prioritas P, L, C, X
                chosen_variant_id = None
                if var_id is not None:
                    # Cek jika varian permintaan tersedia
                    row_req = item_df[item_df['id'] == var_id]
                    if not row_req.empty and row_req.iloc[0]['stock_qty'] >= qty_total:
                        chosen_variant_id = var_id
                if chosen_variant_id is None:
                    for prefix in ['P', 'L', 'C']:
                        sel = variants_ready[variants_ready['name'].str.startswith(prefix)]
                        if not sel.empty:
                            chosen_variant_id = sel['id'].iloc[0]
                            break
                if chosen_variant_id is None:
                    logger.error("Tidak dapat pilih varian untuk produk %s", product_id)
                    return f"Stok varian tidak mencukupi untuk {item.get('product_name')}."
                idx = f"{product_id}|{chosen_variant_id}"
            # Tambah ke order
            logger.debug("Menambahkan paket-item: %s, qty: %s", idx, qty_total)
            print(f"PAKET {item['product_name']:<45} {idx:<15} {qty_total:<15}")

            try:
                resp = add_prod_to_order(order_id, idx, qty_total, access_token=access_token)
                if resp is None:
                    update_status(order_id, "X", access_token=access_token)
                    return "Gagal menambahkan paket ke order, produk habis. Struk di-voidkan."
                logger.debug("Response add paket-item: %s", getattr(resp, 'text', ''))
            except requests.exceptions.HTTPError as http_err:
                logger.error("HTTPError add paket-item: %s", http_err)
                update_status(order_id, "X", access_token=access_token)
                return "Ada kesalahan HTTP saat memasukkan paket ke order. Struk di-voidkan."
            except Exception as e:
                logger.error("Error lain add paket-item: %s", e)
                update_status(order_id, "X", access_token=access_token)
                return f"Ada kesalahan saat memasukkan paket ke order. Error: {e}"
            total_harga_normal += harga * qty_total

        # Setelah semua ditambahkan, update diskon tiap item
        try:
            ord_detail = fetch_order_details(order_id=order_id, access_token=access_token)
            paket_items_added = ord_detail['data']['orderitems'][-len(combo_items):]
            harga_promo = float(combo_details['data']['sell_price_pos']) * qty_pesan
        except Exception as e:
            logger.error("Gagal fetch detail order untuk update diskon: %s", e)
            return "Gagal update diskon paket."
        # Hitung dan update tiap item
        for item in paket_items_added:
            item_id = item['id']
            item_qty = item['qty']
            try:
                item_price = float(item['amount'])
                # Hindari ZeroDivisionError
                if total_harga_normal > 0:
                    item_disc = item_price * ((total_harga_normal - harga_promo) / total_harga_normal)
                else:
                    item_disc = 0.0
            except Exception:
                item_disc = 0.0
            # Ambil fprice bersih angka
            try:
                fprice_str = item.get('fprice', '').replace('.', '')
                price_int = int(float(fprice_str)) if fprice_str else 0
            except Exception:
                price_int = 0
            logger.debug("Update detail order item_id=%s, disc=%s, price=%s", item_id, item_disc, price_int)
            try:
                update_order_detail(
                    order_id=str(order_id),
                    id=str(item_id),
                    disc=str(item_disc),
                    price=str(price_int),
                    qty=str(item_qty),
                    note="Promo Paket",
                    access_token=access_token
                )
            except Exception as e:
                logger.error("Gagal update_order_detail untuk item_id %s: %s", item_id, e)
                return "Gagal update detail paket di order."
        return True

    def handle_order(self, query: str, access_token_dir: str, sudah_bayar: bool = False):
        with open(access_token_dir, "r") as file:
            token_data = json.load(file)
          
        access_token = token_data.get("access_token", "")
        
        logger.debug("Query diterima: %s", query)
        reconfirm_json = self.reconfirm_translator(query)
        logger.debug("Hasil reconfirm: %s", reconfirm_json)
        print("Hasil reconfirm:", reconfirm_json)


        if reconfirm_json.get('fallback'):
            print("[ERROR] Format pesan tidak sesuai:", reconfirm_json['fallback'])
            return reconfirm_json['fallback']
        
        if reconfirm_json.get('pembatalan'):
          print("[DEBUG] Pembatalan order dengan ID:", reconfirm_json['pembatalan'])

          # Kalau masih bukan list setelah semua itu, bungkus jadi list
          if not isinstance(reconfirm_json['pembatalan'], list):
              reconfirm_json['pembatalan'] = [str(reconfirm_json['pembatalan'])]

          # Cek apakah list-nya kosong
          if not reconfirm_json['pembatalan']:
              print("[ERROR] Tidak ada ID yang diberikan untuk pembatalan.")
              return "Tidak ada ID yang diberikan untuk pembatalan."

          # Batalkan order
          for order_id in reconfirm_json['pembatalan']:
              try:
                #   update_status(order_id, "X", access_token)
                  stat_void = void_order(order_id, access_token)
                  if stat_void:
                      print(f"Order {order_id} berhasil di-void.")
                  else:
                      raise ValueError(f"Order {order_id} tidak ditemukan, kemungkinan order ini sudah di-void.")
              except requests.exceptions.HTTPError as http_err:
                  return f"Ada error dari Olsera API dalam membatalkan order {order_id}."
              except Exception as err:
                  return f"Ada kesalahan dalam membatalkan order {order_id}: {err}"

          return f"Order dengan ID {', '.join(reconfirm_json['pembatalan'])} telah dibatalkan."

        # Ubah alamat
        try:
            if reconfirm_json['address'][:4] == "http":
                alamat_cust, longlat_cust, kelurahan, kecamatan, kota, provinsi = resolve_maps_shortlink(reconfirm_json['address'], api_key=self.gmap_api_key)
                distance_and_time = get_travel_distance(self.longlat_toko, longlat_cust, api_key=self.gmap_api_key)
                distance = distance_and_time['distance_meters'] / 1000
                reconfirm_json['distance'] = distance
            else:
                longlat_cust = address_to_latlng(reconfirm_json['address'], api_key=self.gmap_api_key)
                distance_and_time = get_travel_distance(self.longlat_toko, longlat_cust, api_key=self.gmap_api_key)
                distance = distance_and_time['distance_meters'] / 1000
                reconfirm_json['distance'] = distance
            
            if reconfirm_json['distance'] > 45:
                logger.error("Jarak terlalu jauh: %s km", reconfirm_json['distance'])
                return "Maaf, jarak pengiriman terlalu jauh. Silakan hubungi telemarketer untuk bantuan lebih lanjut."
        except Exception as e:
            logger.error("Gagal resolve alamat: %s", reconfirm_json['address'])
            return f"Gagal mengonversi alamat. Pastikan format alamat dalam bentuk link: https://maps.app.goo.gl/XXX."

        try:
            gemini_model = genai.GenerativeModel(
                model_name=self.model_name["flash"],
                system_instruction=task_instructions['notes_prompt'],
            )
            notes = gemini_model.generate_content(query)
            notes_text = getattr(notes, 'text', '') or ''
            logger.debug("Notes dihasilkan: %s", notes_text)
        except Exception as e:
            logger.error("Gagal generate notes: %s", e)
            notes_text = ""

        try:
            kastamer = cek_kastamer(
                nomor_telepon=reconfirm_json['phone_num'],
                access_token=access_token
            )
            
            cust_telp = reconfirm_json['phone_num']

            if kastamer is None:
                cust_id = None
                cust_name = reconfirm_json['cust_name']
            else:
                cust_id = kastamer[0]
                cust_name = kastamer[1]

        except Exception as e:
            logger.error("Gagal cek atau buat customer: %s", e)
            return "Gagal memproses data pelanggan."

        # Create order
        today_str = datetime.now().strftime('%Y-%m-%d')
        try:
            print("Membuat order baru...")
            print(cust_id, cust_name, cust_telp)
            order_id, order_no = create_order(
                order_date=today_str,
                customer_id=cust_id,
                nama_kastamer=cust_name,
                nomor_telepon=cust_telp,
                notes=notes_text,
                access_token=access_token
            )
            logger.debug("Order dibuat: ID=%s, No=%s", order_id, order_no)
            print(f"Order ID: {order_id}, Order No: {order_no}")
        except Exception as e:
            logger.error("Gagal membuat order: %s", e)
            return "Terjadi kesalahan, gagal membuat order baru. Mohon coba lagi."
        
        subsidi_ongkir = is_free_delivery(alamat_cust, self.free_areas)
        ongkir = distance_cost_rule(reconfirm_json['distance'], subsidi_ongkir[0])
        # Add Ongkir
        if ongkir != "Gratis Ongkir" and ongkir != "Subsidi Ongkir 10K":
            reconfirm_json['ordered_products'].append(
                {
                    'tipe': 'Item',
                    'produk': distance_cost_rule(reconfirm_json['distance']),
                    'quantity': 1,
                }
            )

        elif ongkir == "Subsidi Ongkir 10K":
            reconfirm_json['ordered_products'].append(
                {
                    'tipe': 'Item',
                    'produk': "Subsidi Ongkir 10K",
                    'quantity': 1,
                }
            )
        
        else:
            pass

        # print(reconfirm_json)
        print("PESANAN RECONFIRM")
        print("Nama Pelanggan:", reconfirm_json['cust_name'])
        print("Nomor Telepon:", reconfirm_json['phone_num'])
        print("Alamat:", reconfirm_json['address'])
        print("-" * 75)
        print(f"{'JENIS':<5} {'BARANG':<45} {'ID':<20} {'QTY':<10}")
        print("-" * 75)

        # Add products to order
        ordered_products = reconfirm_json.get('ordered_products', [])
        for product in ordered_products:
            # print(f"\n[DEBUG] Memproses produk: {product['produk']} | Tipe: {product['tipe']}")
            tipe = product.get('tipe', '').lower()
            nama_produk = product.get('produk', '')
            qty = product.get('quantity', 0)
            
            if not nama_produk or qty <= 0:
                logger.warning("Produk diabaikan karena nama/qty tidak valid: %s", product)
                continue

            if tipe == 'item':
                success, msg = self._process_item(order_id, nama_produk, qty, access_token)

                if not success:
                    # Jika error di dalam, void entire order dan return
                    update_status(order_id, "X", access_token)
                    return f"Ada kesalahan saat menambahkan item, struk di-void. Error: {msg}"

            elif tipe == 'paket':
                success_or_msg = self._process_paket(order_id, nama_produk, qty, access_token)
                if success_or_msg is not True:
                    # Jika mengembalikan string pesan error, batalkan order
                    update_status(order_id, "X", access_token)
                    return success_or_msg
            
            else:
                print(f"[ERROR] Jenis tidak dikenali. Pastikan untuk memasukkan produk dengan kurung () yang menjelaskan jenis produk, apakah item atau paket. Misal: Hennesey 650 mL (item). Anda memasukkan: {product['tipe']}")
                continue
        
        # Tambahkan diskon
        try:
            self.add_discount(order_id, mode=reconfirm_json['mode_diskon'], access_token=access_token, discount=reconfirm_json['disc'], notes="")
        
        except Exception as e:
            logger.error("Gagal menambahkan diskon: %s", e)
            update_status(order_id, "X", access_token)
            return "Ada kesalahan saat menambahkan diskon. Mohon coba kirim ulang, sementara struk di voidkan."

        # Retrieve order details after adding products
        try:
            order_details = fetch_order_details(order_id, access_token)

        except Exception as e:
            logger.error("Gagal fetch detail order setelah tambah produk: %s", e)
            return "Gagal mengambil detail order."
        
        # Auto add merch
        total_amount = int(float(order_details['data']['total_amount']))
        if total_amount < 100000:
            self._process_item(order_id, "Cup Babe", 1, access_token)
        
        else:
            self._process_item(order_id, "Cup Babe", 2, access_token)
        
        if total_amount > 150000 and total_amount < 250000:
            self._process_paket(order_id, "Merch Babe 1", 1, access_token)
        
        elif total_amount >= 250000:
            self._process_paket(order_id, "Merch Babe 2", 1, access_token)
        
        else:
            pass

        # Proses pembayaran
        status = reconfirm_json.get('status', '').lower()
        if sudah_bayar or status == 'lunas':
            try:
                # Ambil ID metode pembayaran
                payment_modes = list_payment_modes(order_id, access_token)
                # payment_dict: mapping nama pembayaran ke indeks
                jenis = reconfirm_json.get('payment_type', '')
                idx = payment_dict.get(jenis)
                if idx is None or idx >= len(payment_modes):
                    logger.warning("Metode pembayaran '%s' tidak dikenal", jenis)
                    # lanjutkan tanpa bayar atau return error?
                else:
                    payment_id = payment_modes[idx]['id']
                    # total_amount = int(float(order_details['data']['total_amount']))
                    update_payment(
                        order_id=order_id,
                        payment_amount=str(total_amount),
                        payment_date=today_str,
                        payment_mode_id=str(payment_id),
                        access_token=access_token,
                        payment_payee="Kevin Tes API Agent AI",
                        payment_seq="0",
                        payment_currency_id="IDR"
                    )
                    update_status(order_id, "Z", access_token)
                    logger.debug("Order %s ditandai lunas dan status diupdate.", order_id)
            except Exception as e:
                logger.error("Gagal proses pembayaran: %s", e)
                # Tidak membatalkan order karena produk sudah masuk; tergantung kebijakan.
        
        # 8. Cetak struk
        try:
            struk_url = cetak_struk(order_no, cust_telp)
        except Exception as e:
            logger.error("Gagal cetak struk: %s", e)
            struk_url = None

        # 9. Buat invoice teks
        pending_line = "*PENDING ORDER*\n" if status != 'lunas' else ""

        # if reconfirm_json.get('jenis_pengiriman') == 'FD':
        #     max_luncur = (datetime.now() + timedelta(minutes=35)).strftime('%H:%M')
        # elif reconfirm_json.get('jenis_pengiriman') == 'I':
        #     max_luncur = (datetime.now() + timedelta(minutes=25)).strftime('%H:%M')
        # elif reconfirm_json.get('jenis_pengiriman') == 'EX':
        #     max_luncur = (datetime.now() + timedelta(minutes=20)).strftime('%H:%M')

        try:
            max_luncur = estimasi_tiba(reconfirm_json['distance'], reconfirm_json['jenis_pengiriman'], datetime.now())
        except Exception as e:
            max_luncur_menit = int(distance_and_time['duration_seconds'] / 60) + 20
            max_luncur = (datetime.now() + timedelta(minutes=max_luncur_menit)).strftime('%H:%M')

        max_luncur_line = f"MAKSIMAL DILUNCURKAN DARI GUDANG: {max_luncur}" if reconfirm_json['jenis_pengiriman'] == 'FD' else f"ESTIMASI SAMPAI: {max_luncur}"

        total_ftotal = order_details['data'].get('ftotal_amount', '')
        invoice_lines = [
            pending_line.strip(),
            f"Nama: {reconfirm_json.get('cust_name', '')}",
            f"Nomor Telepon: {reconfirm_json.get('phone_num', '')}",
            f"Alamat: {reconfirm_json.get('address', '')}",
            "",
            "",
            max_luncur_line.strip(),
            f"Jarak: {reconfirm_json['distance']:.1f} km ({kelurahan}, {kecamatan.replace('Kecamatan ', '').replace('Kec. ', '').replace('kecamatan', '').replace('kec.', '')})",
            "",
            "",
            "Makasih yaa Cah udah Jajan di Babe!",
            f"Total Jajan: {total_ftotal}",
            f"Cek Jajanmu di sini: {struk_url or 'Gagal mencetak struk. Tolong ulangi.'}",
            "",
            "",
            f"Jenis Pengiriman: {reconfirm_json.get('jenis_pengiriman', '')}",
            f"*NOTES: {reconfirm_json.get('notes') or 'Tidak ada catatan tambahan.'}*",
        ]
        invoice = "\n".join([line for line in invoice_lines if line is not None])
        return invoice
    
    def add_discount(self, order_id, mode, access_token, discount=0, notes=""):
        ord_dtl = fetch_order_details(order_id, access_token)
        order_list = ord_dtl['data'].get('orderitems', [])
        id_order = ord_dtl['data'].get('id', 0)
        total_price = float(ord_dtl['data'].get('total_amount', 0))
        order_list = pd.DataFrame(ord_dtl['data'].get('orderitems', []))
        order_list = order_list[order_list['amount'].astype(float) > 0].reset_index(drop=True)

        for index, row in order_list.iterrows():
            item_id = row['id']
            item_qty = row['qty']
            try:
                # print("COBA UPDATE DISKON")
                # print("HARGA TOTAL", total_price)
                # print("DISKON", discount)
                # print("MODE DISKON", mode)
                item_price = float(row['amount'])
                # Hindari ZeroDivisionError
                if total_price > 0 and mode == 'number':
                    item_disc = float(row['discount']) + item_price * (discount / total_price) 
                
                elif total_price > 0 and mode == 'percentage':
                    item_disc = float(row['discount']) + item_price * discount
                
                else:
                    raise ValueError("Mode diskon tidak dikenali atau total_price nol.")

                # print(f"[DEBUG] item_id={item_id}, item_price={item_price}, row_discount={row['discount']}, discount={discount}, item_disc={item_disc}")
                # print("BERHASIL UPDATE DISKON: ", item_disc)
            except Exception as e:
                print(f"Error calculating discount for item {item_id}: {e}")
                item_disc = 0.0
            # Ambil fprice bersih angka
            try:
                fprice_str = row.get('fprice', '').replace('.', '')
                price_int = int(float(fprice_str)) if fprice_str else 0
            except Exception:
                price_int = 0

            try:
                update_order_detail(
                    order_id=str(id_order),
                    id=str(item_id),
                    disc=str(item_disc),
                    price=str(price_int),
                    qty=str(item_qty),
                    note=notes,
                    access_token=access_token
                )
            except Exception as e:
                print(f"Error updating order detail for item {item_id}: {e}")
                return False, f"Gagal update detail order untuk item {item_id}: {e}"
