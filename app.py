import pandas as pd
import json
import os
import pika
from dotenv import load_dotenv
from modules.llm_call_new import AgentBabe
import google.generativeai as genai
pd.options.mode.chained_assignment = None  

load_dotenv()
genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
gmap_api_key = os.getenv("GMAP_API_KEY")
genai.configure(api_key=genai_api_key)


agent = AgentBabe(df_combo_dir='./product_combos.csv', df_product_dir='./product_items.csv', top_k_retrieve=30, gmap_api_key=gmap_api_key)
credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters(
    host='31.97.106.30',
    port=5679,
    credentials=credentials
)


connection = pika.BlockingConnection(parameters)
channel = connection.channel()


channel.queue_declare(queue='whatsapp_hook_queue', durable=True)
channel.queue_declare(queue='whatsapp_message_queue', durable=True)


def send_reply(from_number, to_number, order_body, mode="launch"):
    if mode == "launch":
        print(f"📥 Pesan diterima dari {to_number}: {order_body}")
        response_message = agent.handle_order(order_body, access_token_dir="./token_cache.json")
    else:
        response_message = "Ada kesalahan dalam memproses pesan."

    payload = {
        "command": "send_message",
        "number": from_number,          
        "number_recipient": to_number,  
        "message": response_message
    }

    channel.basic_publish(
        exchange='',
        routing_key='whatsapp_message_queue',
        body=json.dumps(payload),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print(f"📤 Balasan dikirim ke {to_number}")


def callback(ch, method, properties, body):
    try:
        payload = json.loads(body)
        print("✅ Pesan diterima:")
        print(json.dumps(payload, indent=4))

        if payload.get("type") == "order":
            from_number = payload.get("sessionId")  
            to_number = payload.get("from")
            order_body = payload.get("body", "Pesanan tidak lengkap")

            
            send_reply(from_number, to_number, order_body)

        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("✅ Pesan berhasil diproses dan di-acknowledge.")

    except Exception as e:
        print(f"❌ Error saat proses pesan: {e}")
        
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

if __name__ == "__main__":
    try:
        
        
        channel.basic_qos(prefetch_count=1)

        
        channel.basic_consume(
            queue='whatsapp_hook_queue',
            on_message_callback=callback,
            auto_ack=False 
        )

        print("🔄 Menunggu pesan dari 'whatsapp_hook_queue'...")
        channel.start_consuming()
    except KeyboardInterrupt:
        print("🔴 Proses dihentikan oleh pengguna.")
    finally:
        connection.close()
        print("🔌 Koneksi RabbitMQ ditutup.")
