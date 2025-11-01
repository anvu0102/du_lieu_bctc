import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. IMPORT THƯ VIỆN VNSTOCK ---
try:
    from vnstock import Vnstock
except ImportError:
    st.error("Lỗi: Thư viện 'vnstock' chưa được cài đặt. Vui lòng chạy `pip install vnstock`.")
    st.stop()

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(
    page_title="Phân Tích Dữ Liệu Báo Cáo Tài Chính Việt Nam",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KHAI BÁO CÁC MÃ CỔ PHIẾU VÀ LOẠI BÁO CÁO ---
DEFAULT_STOCKS = ["VNM", "FPT", "HPG", "SSI", "VIC"]
REPORT_TYPES = {
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'income_statement': 'Báo cáo Kết quả Kinh doanh',
    'cash_flow': 'Báo cáo Lưu chuyển Tiền tệ'
}
PERIOD_OPTIONS = {
    'year': 'Theo Năm (Annual)',
    'quarter': 'Theo Quý (Quarterly)'
}
SOURCE_DEFAULT = 'TCBS'

# --- 2. HÀM TẢI DỮ LIỆU TÀI CHÍNH TỪ VNSTOCK ---
@st.cache_data(show_spinner="Đang trích xuất dữ liệu Báo cáo Tài chính...")
def get_financial_data(symbol, period='year', source=SOURCE_DEFAULT):
    """
    Tải Bảng Cân đối Kế toán, Báo cáo KQKD, và Báo cáo Lưu chuyển Tiền tệ
    cho một mã cổ phiếu sử dụng Vnstock.
    """
    st.info(f"Đang tải dữ liệu tài chính cho mã **{symbol}** (Nguồn: {source}, Kỳ: {period})...")
    financial_data = {}
    
    try:
        stock_api = Vnstock().stock(symbol=symbol, source=source)
        
        # Bảng cân đối kế toán
        financial_data['balance_sheet'] = stock_api.finance.balance_sheet(period=period)
        
        # Báo cáo KQKD
        financial_data['income_statement'] = stock_api.finance.income_statement(period=period)
        
        # Báo cáo lưu chuyển tiền tệ
        financial_data['cash_flow'] = stock_api.finance.cash_flow(period=period)

        st.success(f"Tải dữ liệu thành công cho **{symbol}**.")
        return financial_data
        
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu cho **{symbol}**: {e}")
        st.warning("Vui lòng kiểm tra lại mã cổ phiếu và đảm bảo API nguồn dữ liệu đang hoạt động.")
        return None

# --- 3. GIAO DIỆN STREAMLIT ---
st.title("Phân Tích Báo Cáo Tài Chính Cổ Phiếu Việt Nam")
st.markdown("Sử dụng thư viện **`vnstock`** để trích xuất dữ liệu tài chính (Bảng Cân đối Kế toán, Báo cáo KQKD, Lưu chuyển Tiền tệ).")

st.sidebar.header("Tùy Chọn Dữ Liệu")

# Chọn mã cổ phiếu
symbol = st.sidebar.text_input(
    "Nhập Mã Cổ Phiếu (ví dụ: VNM, HPG)",
    value=DEFAULT_STOCKS[0]
).upper()

# Chọn kỳ báo cáo
period = st.sidebar.radio(
    "Chọn Kỳ Báo Cáo:",
    options=list(PERIOD_OPTIONS.keys()),
    format_func=lambda x: PERIOD_OPTIONS[x],
    index=0
)

# Chức năng chính
if symbol:
    
    financial_data = get_financial_data(symbol, period=period, source=SOURCE_DEFAULT)

    if financial_data:
        
        # --- TAB HIỂN THỊ DỮ LIỆU ---
        tabs = st.tabs([f"1. {REPORT_TYPES[key]}" for key in REPORT_TYPES.keys()])
        
        # Hiển thị từng loại báo cáo trong các tab
        for i, (key, name) in enumerate(REPORT_TYPES.items()):
            with tabs[i]:
                st.subheader(f"{name} của {symbol} (Kỳ: {PERIOD_OPTIONS[period]})")
                df = financial_data[key]
                
                if df is not None and not df.empty:
                    # Chuyển đổi cột 'ReportDate' hoặc cột chứa năm/quý sang định dạng ngày/chuỗi để sắp xếp
                    # Giả định cột thời gian là cột đầu tiên sau cột chỉ mục (nếu có)
                    time_col = df.columns[0] 
                    if time_col in ['ReportDate', 'Period']: # Thường là cột đầu tiên
                         df_display = df.sort_values(by=time_col, ascending=False).reset_index(drop=True)
                    else:
                        df_display = df.copy().reset_index(drop=True)

                    st.dataframe(df_display, use_container_width=True)

                    # --- CHỨC NĂNG TẢI VỀ ---
                    @st.cache_data
                    def to_excel(df_to_save):
                        output = BytesIO()
                        # Loại bỏ các ký tự đặc biệt hoặc xử lý cột nếu cần thiết trước khi lưu
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_to_save.to_excel(writer, index=False, sheet_name=name.replace(' ', '_'))
                        return output.getvalue()

                    excel_data = to_excel(df_display)
                    st.download_button(
                        label=f"📥 Tải {name} về Excel (.xlsx)",
                        data=excel_data,
                        file_name=f'{symbol}_{key}_{period}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f'download_{key}'
                    )

                else:
                    st.warning(f"Không tìm thấy hoặc dữ liệu {name} bị trống cho mã **{symbol}**.")

    # --- 4. TRỰC QUAN HÓA (VÍ DỤ: Lợi nhuận ròng) ---
    st.subheader("Trực quan hóa Dữ liệu Quan trọng")

    if financial_data and 'income_statement' in financial_data:
        df_income = financial_data['income_statement']
        
        # Chỉ lấy các cột số
        numeric_cols = df_income.select_dtypes(include=np.number).columns
        chart_cols = ['NetProfit'] if 'NetProfit' in df_income.columns else numeric_cols.tolist()
        
        if chart_cols:
            selected_metric = st.selectbox(
                "Chọn chỉ tiêu cần trực quan hóa từ Báo cáo KQKD:",
                options=chart_cols,
                index=0
            )

            # Lấy cột thời gian (thường là cột đầu tiên, ví dụ: ReportDate)
            time_col = df_income.columns[0] 
            
            # Chuẩn bị dữ liệu để vẽ
            df_chart = df_income[[time_col, selected_metric]].dropna()
            
            if not df_chart.empty:
                # Sắp xếp theo thời gian tăng dần để vẽ biểu đồ đường/cột
                df_chart = df_chart.sort_values(by=time_col, ascending=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(x=df_chart[time_col], y=df_chart[selected_metric], marker='o', ax=ax)
                
                # Biểu đồ cột có thể phù hợp hơn
                # sns.barplot(x=df_chart[time_col], y=df_chart[selected_metric], ax=ax, palette='viridis')

                ax.set_title(f"Xu hướng {selected_metric} của {symbol} ({PERIOD_OPTIONS[period]})", fontsize=16)
                ax.set_xlabel("Kỳ Báo Cáo", fontsize=12)
                ax.set_ylabel(selected_metric, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning(f"Không có dữ liệu hợp lệ cho chỉ tiêu '{selected_metric}' để vẽ biểu đồ.")
        else:
            st.warning("Không tìm thấy các chỉ tiêu định lượng (numeric) trong Báo cáo KQKD để trực quan hóa.")
            
else:
    st.info("Vui lòng nhập Mã Cổ Phiếu để bắt đầu.")