from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import google.generativeai as genai
import re
import json
import pandas as pd
import requests
import ast
from modules.crud_utility import add_prod_to_order, update_payment, fetch_product_combo_details, fetch_order_details, update_order_detail, cek_kastamer, create_order, list_payment_modes, update_status, cetak_struk, fetch_product_item_details, update_order_attr
from datetime import datetime, timedelta

nltk.download('punkt')
nltk.download('punkt_tab')

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
Tuker Voucher: Tumblr (Tidak wajib, jika tidak ada, kosongkan saja; default quantity = 1)
Notes: Es Batu, Sticker 3 (Tidak wajib, jika tidak ada, kosongkan saja)
Pengiriman: FD (Tidak wajib, default "SD", jika tidak disebutkan)

Jika ada permintaan untuk melakukan voiding seperti, "batalkan struk <ID>", "void struk <ID>", atau "batalkan order <ID>", maka kembalikan pesan dengan format:
{
    "pembatalan": "<ID>"
}

Tapi pastikan ada IDnya, jika tidak ada ID, kembalikan pesan dengan format:
{
    "fallback": "Tidak ada ID yang diberikan untuk pembatalan."
}

2. Output harus berupa JSON valid, dengan struktur dan aturan berikut:

{
  "cust_name": <str>,
  "phone_num": <str>,
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
  "jenis_pengiriman":<"FD"|"EX"|"SD"> (default "SD")
  "status": <"Lunas"|"Pending"> (default "Lunas")
}

Keterangan:
- cust_name: Nama pelanggan.
- phone_num: Nomor telepon pelanggan.
- ordered_products: Array objek, satu per produk atau voucher:
  a.) tipe: "Paket" untuk produk berjenis paket/voucher (lihat daftar di bawah), "Item" untuk produk perbiji atau tambahan seperti es batu atau stiker. INGAT!! bahwa "Item" tidak mungkin berisi lebih dari satu produk dalam satu kuantitas. Jadi jika user meminta "Atlas Lychee 2 Botol Promo (Paket)", maka tipe-nya adalah "Paket". Jika user meminta "3 Anggur Merah 500 mL (Item)", maka tipe-nya adalah "Item".
  b.) produk: Nama paket atau item persis seperti di pesan.
  c.) quantity: Jumlah. Jika pelanggan tidak mencantumkan angka, default 1.

- address: Alamat untuk pengiriman.
- payment_type: Harus salah satu dari BCA, BRI, Cash, QRIS, Hutang. Lainnya => Cash.
- notes: Semua permintaan khusus selain ordered_products. Jika pelanggan meminta rokok atau bawain sesuatu, atau layanan di luar delivery alkohol, jadikan notes serta tambahkan item "Nitip ke Jagoane Babe (Item)" dengan quantity 1, kecuali jika minta es batu, cup, atau stiker, maka tambahkan sebagai produk terpisah dengan tipe "Item" dan quantity sesuai permintaan.
- PASTIKAN UNTUK MEMETAKAN INPUT KE TIPE "Item" JIKA ADA KURUNG BERTULISAN (Item) ATAU KE TIPE "Paket" JIKA ADA KURUNG BERTULISAN (Paket). Misal: "3 botol Atlas Lychee 600 mL (Item)" => tipe "Item", quantity 3, "4 paket Atlas Lychee 2 Botol Promo (Paket)" => tipe "Paket", quantity 4.

"jenis_pengiriman": Jenis pengiriman, SD: standard delivery, FD: fast delivery, EX: express. Nilai defaultnya "SD", jika tidak disebutkan
"status": Status pembayaran, "Lunas" jika sudah bayar, "Pending" jika belum bayar. Nilai defaultnya "Lunas", jika tidak disebutkan.

3. Daftar tipe "Paket" tambahan:
- Merch / Merh Babe / Polos XXX (XXX ini angka, misal "Merch Babe 1", "Merh Polos 2". Jadi jika pelanggan meminta "Merch Babe 2", maka tipe-nya adalah "Paket" dan produk-nya adalah "Merch Babe 2", quantity = 1, bukan "Merch Babe", quantity = 2)
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
Nama: [Harus ada]
Nomor Telepon: [Harus ada]
Produk: [Harus ada, sertakan tipe Paket/Item]
Alamat: [Harus ada]
Payment: [Harus ada, BCA/BRI/Cash/QRIS/Hutang]
Tuker Voucher: [Tidak wajib]
Notes: [Tidak wajib]
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
Notes: Es Batu, Sticker 2, nitip rokok

Contoh Output JSON BENAR:
{
  "cust_name": "Budi",
  "phone_num": "08987654321",
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
      "tipe": "Item",
      "produk": "Nitip ke Jagoane Babe",
      "quantity": 1
    }
  ],
  "address": "Jl. Melati No. 5, Bandung",
  "payment_type": "QRIS",
  "notes": "Tambahan stiker 2, nitip rokok",
  "jenis_pengiriman": "SD"
}

Contoh Output JSON SALAH (fallback):
{
  "cust_name": "Budi",
  "phone_num": "08987654321",
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
  "jenis_pengiriman": "SD"
}
'''

item_selection_prompt = '''
Anda adalah sebuah agent yang bertugas untuk memilih indeks data yang paling sesuai dengan prompt yang diberikan.
Cukup jawab dengan nomor indeks data yang paling sesuai dengan prompt yang diberikan.
Tidak perlu menjelaskan apapun, cukup jawab dengan nomor indeks data yang paling sesuai dengan format yang telah ditentukan.
Jika ada produk yang punya tulisan "- I" atau "- O", utamakan "- I" dahulu.

CATATAN: JIKA PRODUK TIDAK SESUAI, JANGAN MENJAWAB APAPUN!
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
'''

merch_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri merch dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri garansi yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

garansi_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri garansi dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri garansi yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

kupon_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri kupon dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri kupon yang paling sesuai dengan permintaan pelanggan dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

voucher_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri voucher dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri voucher yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

komplimen_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri komplimen dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri voucher yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

delivery_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri delivery dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri delivery yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

hadiah_selection_prompt = '''
Kamu adalah AI Agent Kasir untuk Kulkas Babe (@kulkasbabe.id), bertugas memilih satu entri hadiah dari daftar berikut berdasarkan permintaan pelanggan.
Pilih satu entri hadiah yang paling sesuai dengan permintaan pelanggan, harus bener-bener sesuai dan tuliskan saja IDnya langsung, misal 228431 (tipe integer). Tidak perlu penjelasan atau teks tambahan.
'''

notes_prompt = '''
Beri salam hangat kepada pelanggan, tidak usah menggunakan bold atau bahasa terlalu formal.
Sebutkan paket-paket promo yang dipesan. Jawab dengan gaya persis seperti di bawah ini.

Contoh input:
RECONFIRM JAJAN
Nama: Kevin
Nomor Telepon: 085853605806
Produk: 3 botol QRO anggur merah 650 ml (Item); Paket 3 AO Mild (Paket); 1 buah Paket 2 Anggur Hijau MCD + Kawa (Paket); Ongkir 10K (Item)
Alamat: Jl. Melati No. 5, Bandung
Payment: BCA
Notes: Es Batu 3

Contoh output:

"Matur suwun cah, pesenanmu wis kami proses. Paket-paket promo sing mbok pesen Paket 2 Atlas Lychee, Paket 3 Botol Anker Lychee, lan Paket 3 Botol Vibe Lychee yoo."
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
    ):
        self.instructions = instructions
        self.model_name = model_name
        self.df_product_dir = df_product_dir
        self.df_combo_dir = df_combo_dir
        self.top_k_retrieve = top_k_retrieve

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
            model_name="gemini-1.5-flash",
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
            model_name="gemini-1.5-flash",
            system_instruction=self.instructions['reconfirm_translator_prompt'],
        )

        gemini_ans = model.generate_content(message)
        sanitized_response = self.clean_llm_json_output(gemini_ans.text)

        return sanitized_response

    def handle_order(self, query: str, access_token: str, sudah_bayar: bool = False):
        print("[DEBUG] Query diterima:", query)
        reconfirm_json = self.reconfirm_translator(query)
        print("[DEBUG] Hasil reconfirm:", reconfirm_json)
        if reconfirm_json.get('fallback'):
            print("[ERROR] Format pesan tidak sesuai:", reconfirm_json['fallback'])
            return reconfirm_json['fallback']
        
        elif reconfirm_json.get('pembatalan'):
            print("[DEBUG] Pembatalan order dengan ID:", reconfirm_json['pembatalan'])
            update_status(reconfirm_json['pembatalan'], "X", access_token)
            return f"Order dengan ID {reconfirm_json['pembatalan']} telah dibatalkan."

        gemini_model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=task_instructions['notes_prompt'],
        )
        notes = gemini_model.generate_content(query)

        cust_id, cust_telp = cek_kastamer(
            nomor_telepon=reconfirm_json['phone_num'],
            access_token=access_token
        )

        # Create order
        order_id, order_no = create_order(
            order_date=f"{datetime.now().strftime('%Y-%m-%d')}",
            customer_id=cust_id,
            nama_kastamer=reconfirm_json['cust_name'],
            nomor_telepon=cust_telp,
            notes=notes.text,
            access_token=access_token
        )

        print(f"[DEBUG] ID Order berhasil dibuat: {order_id}, Nomor: {order_no}")

        # print(reconfirm_json)
        print("PESANAN RECONFIRM")
        print("Nama Pelanggan:", reconfirm_json['cust_name'])
        print("Nomor Telepon:", reconfirm_json['phone_num'])
        print("Alamat:", reconfirm_json['address'])
        print("-" * 75)
        print(f"{'JENIS':<5} {'BARANG':<45} {'ID':<20} {'QTY':<10}")
        print("-" * 75)

        # Add products to order
        for product in reconfirm_json['ordered_products']:
            # print(f"\n[DEBUG] Memproses produk: {product['produk']} | Tipe: {product['tipe']}")
            if product['tipe'].lower() == 'item':
                df = self.product_df[self.product_df['pos_hidden'] == 0] # Pastikan untuk menunjukkan produk yang tidak tersembunyi di POS.
                idx = self.select_id_by_agent(product['produk'], df, self.instructions['item_selection_prompt'], id_col='id', evaluation_col='name')
                if idx is None:
                    # print(f"[ERROR] Gagal menemukan produk {product['produk']} dalam daftar produk.")
                    continue
                df = df[df['id'] == idx].reset_index(drop=True)
                
                # print("[DEBUG] ID produk terpilih:", idx)

                if df['variants'][0] != '[]':
                    # Reformat to Series
                    variants = ast.literal_eval(df['variants'][0])
                    variants = pd.DataFrame(variants)

                    variants_ready = variants[variants['stock_qty'] > 0]
                    prioritas = ['L', 'P', 'C']

                    # print("[DEBUG] Varian tersedia:")
                    # print(variants_ready)

                    for prefix in prioritas:
                        selected_variant = variants_ready[variants_ready['name'].str.startswith(prefix)]
                        if not selected_variant.empty:
                            variant_name = selected_variant['name'].iloc[0]
                            variant_id = selected_variant['id'].iloc[0]

                            # print(f"ITEM {variant_name:<40} {variant_id:<10}")

                            idx = f"{idx}|{variant_id}"
                            # print(f"[DEBUG] Varian {prefix} terpilih:", variant_name, variant_id)
                            break
                
                # print(f"[DEBUG] Menambahkan produk ke order: {idx} | Qty: {product['quantity']}")
                print(f"ITEM  {df['name'].iloc[0]:<45} {idx:<15} {product['quantity']:<15}")
                try:
                  resp = add_prod_to_order(order_id, idx, product['quantity'], access_token=access_token) # Masih template, belum fungsional
                except requests.exceptions.HTTPError as http_err:
                    print(f"Ada kesalahan pada inputting, struk di void-kan: {http_err} - Response: {resp.text}")
                    update_status(order_id, "X", access_token)
                    return f"Ada kesalahan dalam memasukkan produk ke order: {http_err} - {resp.text}. Struk di-voidkan."
                except Exception as err:
                    print(f"Ada kesalahan lain pada inputting, struk di void-kan: {err}")
                    update_status(order_id, "X", access_token)
                    return f"Ada kesalahan dalam memasukkan produk ke order: {http_err}. Struk di-voidkan."

            elif product['tipe'].lower() == 'paket':
                df = self.combo_df[self.combo_df['pos_hidden'] == 0] # Pastikan untuk menunjukkan paket yang tidak tersembunyi di POS.
                task_instruction = self.instructions['combo_selection_prompt']

                product_name_lower = product['produk'].strip().lower()

                # Subset berdasarkan awalan:
                if product_name_lower.startswith('merch') or product_name_lower.startswith('mer') or 'merch' in product_name_lower:
                    mask = df['name'].str.lower().str.contains('merch|merh')
                    df = df[mask]
                    task_instruction = self.instructions['merch_selection_prompt']

                elif product_name_lower.startswith(('babe garansiin','garansi','garan')):
                    df = df[df['name'] == 'Babe Garansi-in !!!']
                    task_instruction = self.instructions['garansi_selection_prompt']

                elif 'kupon' in product_name_lower:
                    mask = df['name'].str.lower().str.contains('kupon')
                    df = df[mask]
                    task_instruction = self.instructions['kupon_selection_prompt']
                
                elif 'voucher' in product_name_lower:
                    mask = df['name'].str.lower().str.contains('voucher')
                    df = df[mask]
                    task_instruction = self.instructions['voucher_selection_prompt']

                elif product_name_lower.startswith(('komplimen', 'komp')):
                    mask = df['name'].str.lower().str.startswith(('komplimen','komp'))
                    df = df[mask]
                    task_instruction = self.instructions['komplimen_selection_prompt']

                elif 'delivery' in product_name_lower:
                    mask = df['name'].str.lower().str.contains('delivery')
                    df = df[mask]
                    task_instruction = self.instructions['delivery_selection_prompt']

                elif product_name_lower.startswith('hadiah'):
                    df = df[df['name'].str.lower().str.startswith('hadiah')]
                    task_instruction = self.instructions['hadiah_selection_prompt']

                if len(df) < 2:
                    idx = df['id'].iloc[0] if not df.empty else None
                else:
                    idx = self.select_id_by_agent(product['produk'], df, task_instruction, id_col='id', evaluation_col='name')
                if idx is None:
                    print(f"[ERROR] Gagal menemukan produk {product['produk']} dalam daftar produk.")
                    continue

                # Retrieve data dari combo detail.
                # print(f"[DEBUG] ID paket terpilih:", idx)
                combo_details = fetch_product_combo_details(idx, access_token)
                # print("[DEBUG] Detail paket:", combo_details)
                combo_items = combo_details['data']['items']['data']

                total_harga_normal = 0

                # Tambahkan dulu produk paket ke order
                for item in combo_items:
                    qty = item['qty'] * product['quantity']  # Mengalikan qty paket dengan qty yang dipesan

                    item_details = fetch_product_item_details(item['product_id'], access_token=access_token)
                    product_data = item_details.get('data', {})

                    if len(product_data['variant']) == 0:
                        idx = f"{item['product_id']}"
                        harga = f"{product_data['sell_price_pos']}"
                    else:
                        item_details = item_details['data']['variant']
                        item_details_df = pd.DataFrame(item_details)
                        harga = item_details_df['sell_price_pos'].iloc[0] if not item_details_df.empty else None

                        if 'stock_qty' not in item_details_df.columns:
                            print(f"[ERROR] Data stok tidak ditemukan untuk {item['product_name']} (ID: {item['product_id']}). Struk akan di-void.")
                            update_status(order_id, "X", access_token)
                            return f"Data stok tidak ditemukan untuk {item['product_name']} dalam database. Struk di-voidkan."

                        variants_ready = item_details_df[item_details_df['stock_qty'] >= qty]  # Lebih baik pakai >=

                        if variants_ready.empty:
                            print(f"[ERROR] Tidak ada stok yang cukup untuk {item['product_name']} (ID: {item['product_id']}). Struk akan di-void.")
                            update_status(order_id, "X", access_token)
                            return f"Maaf, stok tidak cukup untuk {item['product_name']} dalam database. Struk di-voidkan."

                        
                        prioritas = ['P', 'L', 'C', 'X']

                        if variants_ready[variants_ready['id'] == item['product_variant_id']]['stock_qty'].iloc[0] < qty:
                            # Jika varian yang diminta tidak tersedia, cari varian lain
                            print(f"[DEBUG] Varian {item['product_variant_id']} tidak tersedia untuk {item['product_name']}. Mencari varian lain...")
                            for prefix in prioritas:
                                selected_variant = variants_ready[variants_ready['name'].str.startswith(prefix)]

                                if not selected_variant.empty:
                                    variant_name = selected_variant['name'].iloc[0]
                                    variant_id = selected_variant['id'].iloc[0]

                                    # print(f"ITEM {variant_name:<40} {variant_id:<10}")

                                    idx = f"{item_details_df['product_id'].iloc[0]}|{variant_id}"
                                    # print(f"[DEBUG] Varian {prefix} terpilih:", variant_name, variant_id)
                                    break
                        else:
                            # Jika varian yang diminta tersedia, gunakan varian tersebut
                            idx = f"{item['product_id']}|{item['product_variant_id']}"

                    print(f"PAKET {item['product_name']:<45} {idx:<15} {qty:<15}")
                    total_harga_normal += float(harga) * qty

                    try:
                        resp = add_prod_to_order(order_id, idx, qty, access_token=access_token) # Masih template, belum fungsional
                    except requests.exceptions.HTTPError as http_err:
                        print(f"Ada kesalahan pada inputting, struk di void-kan: {http_err} - Response: {resp.text}")
                        update_status(order_id, "X", access_token)
                        return f"Ada kesalahan dalam memasukkan produk ke order: {http_err} - {resp.text}. Struk di-voidkan."
                    except Exception as err:
                        print(f"Ada kesalahan lain pada inputting, struk di void-kan: {err}")
                        update_status(order_id, "X", access_token)
                        return f"Ada kesalahan dalam memasukkan produk ke order: {http_err}. Struk di-voidkan."

                # Baru setelah semua item paket ditambahkan, hitung diskon masing-masing item
                ord_detail = fetch_order_details(order_id=order_id, access_token=access_token)
                paket_items = ord_detail['data']['orderitems'][-len(combo_items):]
                harga_promo = float(combo_details['data']['sell_price_pos']) * product['quantity']

                for item in paket_items:
                    item_id = item['id']
                    item_qty = item['qty']
                    item_price = int(float(item['amount']))
                    try:
                      item_disc = (item_price * ((total_harga_normal - harga_promo) / total_harga_normal))
                    except:
                      item_disc = 0
                    # print(f"[DEBUG] Memperbarui detail order ID {item_id} dengan diskon {item_disc} dan harga {harga_promo}")
                    update_order_detail(
                        order_id=str(order_id),
                        id=str(item_id),
                        disc=f"{item_disc}",
                        price=f"{int(float(item['fprice'].replace('.', '')))}",
                        qty=str(item_qty),
                        note="Promo Paket",
                        access_token=access_token
                    )                
                
            else:
                print(f"[ERROR] Jenis tidak dikenali. Pastikan untuk memasukkan produk dengan kurung () yang menjelaskan jenis produk, apakah item atau paket. Misal: Hennesey 650 mL (item). Anda memasukkan: {product['tipe']}")
                continue

        # Lanjut ke proses transaksi dan close order. tbd.
        order_details = fetch_order_details(order_id, access_token)
        if sudah_bayar or reconfirm_json['status'].lower() == 'lunas':
            payment_id = list_payment_modes(order_id, access_token)[payment_dict[reconfirm_json['payment_type']]]['id']
            update_payment(
                order_id=order_id,
                payment_amount=f"{int(float(order_details['data']['total_amount']))}",
                payment_date=f"{datetime.now().strftime('%Y-%m-%d')}",
                payment_mode_id=f"{payment_id}", 
                access_token=access_token,
                payment_payee="Kevin Tes API Agent AI",
                payment_seq="0",
                payment_currency_id="IDR"
            )

            update_status(order_id, "Z", access_token)

        struk = cetak_struk(order_no, cust_telp)

        pending_line = "*PENDING ORDER*\n" if not reconfirm_json['status'].lower() == 'lunas' else ""

        invoice = f'''{pending_line}Nama: {reconfirm_json['cust_name']}
Nomor Telepon: {reconfirm_json['phone_num']}
Alamat: {reconfirm_json['address']}
Order ID: {order_id} (Gunakan ID ini untuk melakukan voiding jika ada kesalahan atau pengembalian)
Resi: {order_no}

*MAKSIMAL DILUNCURKAN DARI GUDANG*: {(datetime.now() + timedelta(minutes=35)).strftime('%H:%M')}

Jarak: XXX km 

Thank you for shopping with Kulkas Babe!
Total Order: {order_details['data']['ftotal_amount']}
View Receipt: {struk}

Jenis Pengiriman: {reconfirm_json['jenis_pengiriman']}

Notes: {reconfirm_json['notes'] if reconfirm_json['notes'] else "Tidak ada catatan tambahan."}
'''

        if struk is None:
            struk = "Ada kesalahan dalam mencetak struk. Tolong kirim ulang."
            return struk
        return invoice
