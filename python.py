# Sửa đổi phần đầu của file, đảm bảo các imports cần thiết:
import streamlit as st
import pandas as pd
from io import BytesIO
# ... (các imports khác)
import zipfile
import io
try:
    from fpdf import FPDF
except ImportError:
    st.error("Lỗi: Thư viện 'fpdf2' chưa được cài đặt. Vui lòng chạy `pip install fpdf2`.")
    st.stop()


# ... (Giữ nguyên các hàm khác)

# --- HÀM HỖ TRỢ CHUYỂN DỮ LIỆU SANG ĐỊNH DẠNG PDF (SỬA LỖI UNICODE) ---
def create_pdf_content(symbol, financial_data, period):
    """Chuyển đổi dữ liệu 3 báo cáo thành một file PDF duy nhất."""

    # Tên font tùy chỉnh. LƯU Ý: Phải có file font này (VD: DejaVuSansCondensed.ttf)
    # Nếu đang chạy trên môi trường không có font, đây sẽ là điểm thất bại tiếp theo.
    # Vì lý do đơn giản hóa, tôi sẽ sử dụng font "Arial" nhưng bật uni=True.
    # Trong môi trường thực tế, KHÔNG NÊN DÙNG FONT CHUẨN CỦA FPDF MÀ KHÔNG NHÚNG FONT HỖ TRỢ.
    # Tuy nhiên, do không thể nhúng file font vào code, tôi sẽ dùng cách đơn giản hóa:
    FONT_NAME = 'arial'
    FONT_PATH = 'arial.ttf' # Cần phải có file font này trong thư mục
    
    # Nếu không có file font, hãy thử sử dụng font mặc định và xem môi trường có may mắn hỗ trợ Unicode không.
    # Để chắc chắn, tôi sẽ thêm logic kiểm tra file font.
    
    class PDF(FPDF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                # Thử thêm font hỗ trợ Unicode (Giả định file font đã tồn tại)
                self.add_font(FONT_NAME, '', FONT_PATH, uni=True)
                self.set_font(FONT_NAME, '', 10)
            except Exception:
                # Nếu không tìm thấy font DejaVu, sử dụng font Arial mặc định (và chấp nhận không hỗ trợ Tiếng Việt)
                self.set_font('Arial', '', 10)
                st.warning("Không tìm thấy file font hỗ trợ Unicode. PDF có thể bị lỗi font Tiếng Việt.")

        def header(self):
            # Sử dụng font đã set trong __init__
            self.set_font(self.font_family, 'B', 12)
            self.cell(0, 10, f'BÁO CÁO TÀI CHÍNH {symbol}', 0, 1, 'C')
            self.set_font(self.font_family, '', 10)
            self.cell(0, 5, f'Kỳ: {PERIOD_OPTIONS[period]}', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font(self.font_family, 'I', 8)
            self.cell(0, 10, f'Trang {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font(self.font_family, 'B', 12)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 8, title, 0, 1, 'L', 1)
            self.ln(2)

        def df_to_table(self, df, title):
            self.add_page(orientation='L')
            self.chapter_title(title)
            
            df_temp = df.copy()
            if df_temp.index.names is not None and len(df_temp.index.names) > 0:
                df_temp = df_temp.reset_index(drop=False)

            sort_col = 'id' if 'id' in df_temp.columns else ('ReportDate' if 'ReportDate' in df_temp.columns else df_temp.columns[0])
            if sort_col in df_temp.columns:
                df_temp = df_temp.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
            
            # Cấu hình bảng
            col_width = 270 / len(df_temp.columns)
            row_height = 7
            
            # Header
            self.set_font(self.font_family, 'B', 8)
            for col in df_temp.columns:
                self.cell(col_width, row_height, str(col)[:15], 1, 0, 'C')
            self.ln(row_height)
            
            # Data
            self.set_font(self.font_family, '', 8)
            for index, row in df_temp.iterrows():
                for item in row:
                    try:
                        text = f"{item:,.0f}" if isinstance(item, (int, float, np.number)) else str(item)
                    except ValueError:
                         text = str(item)
                    
                    text = text[:15]
                    self.cell(col_width, row_height, text, 1, 0, 'R')
                self.ln(row_height)
            self.ln(5)

    pdf = PDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()

    has_data = False
    for key, name in REPORT_TYPES.items():
        df = financial_data.get(key)
        
        if df is not None and not df.empty:
            pdf.df_to_table(df, name)
            has_data = True
        else:
            pdf.set_font(pdf.font_family, 'I', 10)
            pdf.cell(0, 10, f'--- Dữ liệu {name} bị trống ---', 0, 1)

    if has_data:
        # FPDF với uni=True xuất ra bytes, không cần encode
        return pdf.output(dest='S')
    return None

# ... (Giữ nguyên các hàm ZIP, thống kê, AI)

# --- PHẦN GIAO DIỆN PHÂN TÍCH DANH SÁCH CỔ PHIẾU (Thay đổi logic hiển thị nút) ---
# ...
elif analysis_mode == 'Phân tích Danh sách Cổ phiếu':
    
    st.sidebar.subheader("Danh sách Mã Cổ phiếu")
    
    stock_list_input = st.sidebar.text_area(
        "Nhập danh sách Mã Cổ phiếu, cách nhau bởi dấu phẩy, khoảng trắng hoặc xuống dòng:",
        value=", ".join(DEFAULT_STOCK_LIST),
        height=150
    )
    
    # Chuẩn hóa và lọc danh sách mã
    stock_list_raw = [s.strip().upper() for s in stock_list_input.replace('\n', ',').replace(' ', ',').split(',') if s.strip()]
    stock_list = list(set(stock_list_raw))
    
    if stock_list:
        
        st.subheader(f"📥 Tải Báo cáo Tài chính cho Danh sách Cổ phiếu ({len(stock_list)} mã)")
        st.info(f"Các mã sẽ được tải: **{', '.join(stock_list)}** (Kỳ: {PERIOD_OPTIONS[period]})")

        if st.button(f"🔍 Tải Dữ liệu Báo cáo Tài chính cho {len(stock_list)} Mã"):
            
            all_financial_data = get_all_financial_data(stock_list, period=period, source=SOURCE_DEFAULT)

            if all_financial_data:
                st.success(f"Đã tải thành công dữ liệu cho {len(all_financial_data)} mã.")
                st.markdown("---")
                st.subheader("Hoàn tất: Tải File ZIP Tổng hợp")
                
                col1, col2, col3 = st.columns(3)
                
                # --- TẠO VÀ TẢI FILE ZIP EXCEL ---
                with col1:
                    with st.spinner('Đang nén báo cáo thành file ZIP (Excel)...'):
                        zip_excel_bytes = create_zip_file_excel(all_financial_data, period)

                    if zip_excel_bytes and len(zip_excel_bytes) > len(all_financial_data) * 100:
                        st.download_button(
                            label="📦 Tải TẤT CẢ Báo cáo (Định dạng Excel)",
                            data=zip_excel_bytes,
                            file_name=f'Bao_cao_tai_chinh_DS_{PERIOD_OPTIONS[period]}.zip',
                            mime='application/zip',
                            key='download_all_zip_excel',
                            help="Tải về một file ZIP chứa các file Excel (3 sheets/mã)."
                        )
                        st.caption("Mỗi mã cổ phiếu là 1 file Excel (3 sheets).")
                    else:
                        st.warning("Không thể tạo file ZIP Excel hoặc dữ liệu rỗng.")
                
                # --- TẠO VÀ TẢI FILE ZIP TXT ---
                with col2:
                    with st.spinner('Đang nén báo cáo thành file ZIP (TXT)...'):
                        zip_txt_bytes = create_zip_file_txt(all_financial_data, period)

                    if zip_txt_bytes and len(zip_txt_bytes) > len(all_financial_data) * 100:
                        st.download_button(
                            label="📄 Tải TẤT CẢ Báo cáo (Định dạng TXT)",
                            data=zip_txt_bytes,
                            file_name=f'Bao_cao_tai_chinh_DS_{PERIOD_OPTIONS[period]}_TXT.zip',
                            mime='application/zip',
                            key='download_all_zip_txt',
                            help="Tải về một file ZIP chứa các file TXT (nối 3 báo cáo/mã)."
                        )
                        st.caption("Mỗi mã cổ phiếu là 1 file TXT (3 báo cáo gộp).")
                    else:
                        st.warning("Không thể tạo file ZIP TXT hoặc dữ liệu rỗng.")

                # --- TẠO VÀ TẢI FILE ZIP PDF (MỚI) ---
                with col3:
                    st.warning("Tính năng PDF yêu cầu nhúng file font Tiếng Việt (DejaVuSansCondensed.ttf) vào ứng dụng.")
                    with st.spinner('Đang nén báo cáo thành file ZIP (PDF)...'):
                        zip_pdf_bytes = create_zip_file_pdf(all_financial_data, period)

                    if zip_pdf_bytes and len(zip_pdf_bytes) > len(all_financial_data) * 100:
                        st.download_button(
                            label="📑 Tải TẤT CẢ Báo cáo (Định dạng PDF)",
                            data=zip_pdf_bytes,
                            file_name=f'Bao_cao_tai_chinh_DS_{PERIOD_OPTIONS[period]}_PDF.zip',
                            mime='application/zip',
                            key='download_all_zip_pdf',
                            help="Tải về một file ZIP chứa các file PDF (nối 3 báo cáo/mã)."
                        )
                        st.caption("Mỗi mã cổ phiếu là 1 file PDF (3 báo cáo gộp).")
                    else:
                        st.error("Không thể tạo file ZIP PDF. **Kiểm tra file font**. (Xem cảnh báo phía trên).")
                
            else:
                st.warning("Không có dữ liệu nào được tải thành công. Vui lòng kiểm tra lại các mã cổ phiếu.")
    
    else:
        st.info("Vui lòng nhập một danh sách Mã Cổ phiếu hợp lệ để bắt đầu.")
