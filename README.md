# SME Tax & Business RAG Assistant

แชตบอตภาษาไทยสำหรับค้นคำตอบเบื้องต้นเรื่องภาษีและการจดทะเบียนธุรกิจ SME โดยใช้เอกสารใน `data/` ร่วมกับ Google Gemini, FAISS และ Streamlit

## สถานะ

โปรเจกต์นี้สำหรับพัฒนาและทดสอบบนเครื่องส่วนตัวเท่านั้น ยังไม่มีการ deploy ขึ้นบริการออนไลน์

## เริ่มใช้งานบน Windows PowerShell

เปิด PowerShell ที่โฟลเดอร์โปรเจกต์ แล้วรัน:

```powershell
.\.venv\Scripts\Activate.ps1
```

ถ้ายังไม่มี virtual environment ให้สร้างก่อน:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

เมื่อต้องการติดตั้งหรืออัปเดตไลบรารี ให้รัน:

```powershell
python -m pip install -r requirements.txt
```

ตั้งค่า Google AI Studio API key เฉพาะบนเครื่องนี้ใน `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "ใส่คีย์ส่วนตัวของคุณที่นี่"
```

หรือกำหนด environment variable `GOOGLE_API_KEY` ใน terminal ที่ใช้รันแอป จากนั้นเริ่มแอปด้วย:

```powershell
streamlit run app.py
```

เมื่อใช้งานเสร็จ ออกจาก virtual environment ด้วย `deactivate`

## ความปลอดภัยของคีย์

ห้ามใส่ API key จริงในไฟล์ที่ commit หรือ push ไป GitHub ไฟล์ `.streamlit/secrets.toml`, `.env` และ `.env.*` ถูกระบุไว้ใน `.gitignore` แต่ควรตรวจสอบ `git status` และ diff ทุกครั้งก่อน commit หากเคย push คีย์จริงขึ้น GitHub ให้เพิกถอนคีย์นั้นและสร้างคีย์ใหม่

## ข้อจำกัดของข้อมูล

คำตอบควรอ้างอิงเฉพาะเอกสารใน `data/` หากเอกสารไม่เพียงพอ แอปควรตอบว่า `ไม่พบข้อมูลในเอกสารอ้างอิง` เอกสารเป็นข้อมูลประกอบการพัฒนา ไม่ใช่คำปรึกษาทางกฎหมายหรือภาษี ควรตรวจสอบหลักเกณฑ์และกำหนดเวลาล่าสุดกับหน่วยงานทางการก่อนนำไปใช้จริง
