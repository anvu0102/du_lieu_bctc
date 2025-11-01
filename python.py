import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pandas.api.types import is_numeric_dtype

# --- 1. IMPORT THƯ VIỆN BỔ SUNG CHO GEMINI AI ---
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    st.error("Lỗi: Thư viện 'google-genai' chưa được cài đặt. Vui lòng chạy `pip install google-genai`.")
    st.stop()
    
try:
    from vnstock import Vnstock
except ImportError:
    st.error("Lỗi: Thư viện 'vnstock' chưa được cài đặt. Vui lòng chạy `pip install vnstock`.")
    st.stop()

# --- SỬA LỖI ATTRIBUTEERROR ---
# Thay đổi cách import SettingWithCopyWarning để tương thích với Pandas mới
try:
    from pandas.errors import SettingWithCopyWarning
    warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
except ImportError:
    pass 
except AttributeError:
    pass


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
    'year': 'Theo Năm',
    'quarter': 'Theo Quý'
}
SOURCE_DEFAULT = 'TCBS'


# --- HÀM TẢI DỮ LIỆU TÀI CHÍNH TỪ VNSTOCK ---
@st.cache_data(show_spinner="Đang trích xuất dữ liệu Báo cáo Tài chính...")
def get_financial_data(symbol, period='year', source=SOURCE_DEFAULT):
    """
    Tải Bảng Cân đối Kế toán, Báo cáo KQKD, và Báo cáo Lưu chuyển Tiền tệ
    cho một mã cổ phiếu sử dụng Vnstock.
    """
    st.info(f"Đang tải dữ liệu tài chính cho mã **{symbol}** (Nguồn: VCI, Kỳ: {period})...")
    financial_data = {}
    
    try:
        stock_api = Vnstock().stock(symbol=symbol, source=source)
        
        financial_data['balance_sheet'] = stock_api.finance.balance_sheet(period=period)
        financial_data['income_statement'] = stock_api.finance.income_statement(period=period)
        financial_data['cash_flow'] = stock_api.finance.cash_flow(period=period)

        st.success(f"Tải dữ liệu thành công cho **{symbol}**.")
        return financial_data
        
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu cho **{symbol}**: {e}")
        st.warning("Vui lòng kiểm tra lại mã cổ phiếu và đảm bảo API nguồn dữ liệu đang hoạt động.")
        return None

# --- HÀM HỖ TRỢ TẠO FILE EXCEL ---
@st.cache_data
def to_excel(df_to_save, name):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sheet_name = name.replace(' ', '_').replace('/', '_').strip()[:30]
        df_to_save.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# --- HÀM TÍNH TOÁN THỐNG KÊ MÔ TẢ (CHO TÀI CHÍNH) ---
def calculate_descriptive_stats(df, report_name):
    """Tính toán thống kê mô tả chi tiết cho các chỉ số tài chính."""
    stats_list = []
    
    df_temp = df.copy()
    if df_temp.index.names is not None and len(df_temp.index.names) > 0:
        df_temp = df_temp.reset_index(drop=False)

    numeric_cols = [col for col in df_temp.columns if is_numeric_dtype(df_temp[col])]
    
    # Tìm cột thời gian linh hoạt
    time_col = 'id'
    if 'id' not in df_temp.columns:
        if 'ReportDate' in df_temp.columns:
            time_col = 'ReportDate'
        elif 'Period' in df_temp.columns:
            time_col = 'Period'
        else:
            time_col = df_temp.columns[0] # Dự phòng

    for col in numeric_cols:
        series = df_temp[col].dropna()
        if series.empty:
            stats_list.append({
                'Chỉ tiêu': col, 'Trung bình (Mean)': 'N/A', 'Độ lệch chuẩn (Std Dev)': 'N/A', 
                'Giá trị nhỏ nhất (Min)': 'N/A', 'Kỳ Min': 'N/A',
                'Giá trị lớn nhất (Max)': 'N/A', 'Kỳ Max': 'N/A',
                'Trung vị (Median)': 'N/A', 'Hệ số biến thiên (CV, %)' : 'N/A'
            })
            continue

        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        median_val = series.median()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

        try:
            df_sorted = df_temp.sort_values(by=time_col)
            
            period_min = df_sorted.loc[df_sorted[col] == min_val, time_col].iloc[0]
            period_max = df_sorted.loc[df_sorted[col] == max_val, time_col].iloc[0]
        except Exception:
            period_min, period_max = 'N/A', 'N/A'

        stats_list.append({
            'Chỉ tiêu': col,
            'Trung bình (Mean)': f"{mean_val:,.0f}", 
            'Độ lệch chuẩn (Std Dev)': f"{std_val:,.0f}",
            'Giá trị nhỏ nhất (Min)': f"{min_val:,.0f}",
            'Kỳ Min': period_min,
            'Giá trị lớn nhất (Max)': f"{max_val:,.0f}",
            'Kỳ Max': period_max,
            'Trung vị (Median)': f"{median_val:,.0f}",
            'Hệ số biến thiên (CV, %)': f"{cv:,.2f}%" if not np.isnan(cv) else 'N/A'
        })

    return pd.DataFrame(stats_list)

# --- HÀM GỌI API GEMINI ---
def get_ai_analysis(stats_df_income, stats_df_balance, symbol, period, api_key):
    """Gửi bảng thống kê đến Gemini để phân tích Báo cáo Tài chính."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        income_markdown = stats_df_income.to_markdown(index=False)
        balance_markdown = stats_df_balance.to_markdown(index=False)

        prompt = f"""
        Bạn là một Chuyên gia Phân tích Tài chính hàng đầu. Nhiệm vụ của bạn là phân tích tình hình kinh doanh và sức khỏe tài chính của công ty {symbol} dựa trên dữ liệu báo cáo tài chính {period} (theo Năm/Quý) trong giai đoạn đã được cung cấp.

        Dưới đây là Bảng Thống kê Mô tả cho các chỉ tiêu quan trọng:

        ### Bảng 1: Thống kê Báo cáo Kết quả Kinh doanh (Tập trung vào Hiệu suất)
        {income_markdown}

        ### Bảng 2: Thống kê Bảng Cân đối Kế toán (Tập trung vào Cấu trúc Tài sản & Nguồn vốn)
        {balance_markdown}
        
        Dựa trên hai bảng thống kê trên, hãy viết một báo cáo phân tích tổng hợp (khoảng 4-6 đoạn) bằng tiếng Việt.
        1.  **Đánh giá Tăng trưởng & Ổn định Doanh thu/Lợi nhuận:** Phân tích Trung bình, Tối đa/Tối thiểu, và đặc biệt là **Hệ số biến thiên (CV)** của Doanh thu/Lợi nhuận. CV cao cho thấy sự bất ổn trong hoạt động kinh doanh.
        2.  **Đánh giá Cấu trúc Tài sản & Nợ:** Phân tích xu hướng Tổng tài sản, Nợ phải trả và Vốn chủ sở hữu. Nhận xét về rủi ro tài chính (tỷ trọng nợ).
        3.  **Nhận xét Khác:** Tổng hợp các điểm mạnh, điểm yếu nổi bật trong giai đoạn phân tích.
        
        Hãy trình bày báo cáo một cách chuyên nghiệp, dễ đọc và tập trung vào các con số quan trọng.
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- 4. GIAO DIỆN STREAMLIT CHÍNH ---
st.title("📈 Phân Tích Báo Cáo Tài Chính Cổ Phiếu Việt Nam")
st.markdown("Sử dụng thư viện **`vnstock`** để trích xuất dữ liệu tài chính.")

st.sidebar.header("Tùy Chọn Dữ Liệu")

symbol = st.sidebar.text_input(
    "Nhập Mã Cổ Phiếu (ví dụ: VNM, HPG)",
    value=DEFAULT_STOCKS[0]
).upper()

period = st.sidebar.radio(
    "Chọn Kỳ Báo Cáo:",
    options=list(PERIOD_OPTIONS.keys()),
    format_func=lambda x: PERIOD_OPTIONS[x],
    index=0
)

# Thêm Khóa API cho AI
st.sidebar.header("Cấu hình AI (Tùy chọn)")
api_key = st.sidebar.text_input("Nhập GEMINI_API_KEY", type="password")
st.sidebar.caption("Sử dụng Khóa API của bạn để kích hoạt Phân tích AI.")


if symbol:
    
    financial_data = get_financial_data(symbol, period=period, source=SOURCE_DEFAULT)

    if financial_data:
        
        # --- TAB HIỂN THỊ DỮ LIỆU ---
        tab_names = [f"{i+1}. {REPORT_TYPES[key]}" for i, key in enumerate(REPORT_TYPES.keys())]
        tab_names.extend(["4. Thống kê Mô tả", "5. Trực quan hóa", "6. Phân tích AI"])
        
        tabs = st.tabs(tab_names)
        
        stats_dfs = {}

        report_keys = list(REPORT_TYPES.keys())
        for i, key in enumerate(report_keys):
            name = REPORT_TYPES[key]
            with tabs[i]:
                st.subheader(f"{name} của {symbol} (Kỳ: {PERIOD_OPTIONS[period]})")
                
                df = financial_data[key].copy() 
                
                if df is not None and not df.empty:
                    if df.index.names is not None and len(df.index.names) > 0:
                        df = df.reset_index(drop=False)
                        
                    # Sắp xếp hiển thị
                    sort_col = 'id' if 'id' in df.columns else ('ReportDate' if 'ReportDate' in df.columns else df.columns[0])
                    
                    df_display = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

                    st.dataframe(df_display, use_container_width=True)

                    stats_dfs[key] = calculate_descriptive_stats(df, name)

                    excel_data = to_excel(df_display, name)
                    st.download_button(
                        label=f"📥 Tải {name} về Excel (.xlsx)",
                        data=excel_data,
                        file_name=f'{symbol}_{key}_{period}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f'download_{key}'
                    )

                else:
                    st.warning(f"Không tìm thấy hoặc dữ liệu {name} bị trống cho mã **{symbol}**.")

        # --- TAB THỐNG KÊ MÔ TẢ ---
        with tabs[3]: 
            st.subheader(f"Thống kê Mô tả Báo cáo Tài chính {symbol}")
            
            if stats_dfs:
                for key, df_stats in stats_dfs.items():
                    st.markdown(f"### {REPORT_TYPES[key]}")
                    st.dataframe(df_stats, use_container_width=True)
            else:
                st.info("Không có dữ liệu thống kê để hiển thị.")
                
            st.caption("""
            **Giải thích:** **Độ lệch chuẩn** và **Hệ số biến thiên** (CV) càng cao cho thấy mức độ biến động/bất ổn của chỉ số trong giai đoạn càng lớn.
            Giá trị được làm tròn.
            """)

        # --- TAB TRỰC QUAN HÓA ---
        with tabs[4]: 
            st.subheader("📊 Trực quan hóa Xu hướng Quan trọng (Báo cáo KQKD)")

            if 'income_statement' in financial_data:
                df_income = financial_data['income_statement'].copy()
                
                if df_income.index.names is not None and len(df_income.index.names) > 0:
                    df_income = df_income.reset_index(drop=False) 

                numeric_cols = df_income.select_dtypes(include=np.number).columns.tolist()
                
                default_metrics = ['NetProfit', 'Revenue', 'GrossProfit']
                chart_cols = [col for col in default_metrics if col in numeric_cols]
                chart_cols.extend([col for col in numeric_cols if col not in chart_cols])
                
                # Sửa lỗi: Tìm cột thời gian linh hoạt
                time_col_for_chart = 'period'

                if chart_cols and time_col_for_chart:
                    selected_metric = st.selectbox(
                        "Chọn chỉ tiêu cần trực quan hóa từ Báo cáo KQKD:",
                        options=chart_cols,
                        index=chart_cols.index('NetProfit') if 'NetProfit' in chart_cols else 0
                    )
                    
                    df_chart = df_income[[time_col_for_chart, selected_metric]].dropna()
                    
                    if not df_chart.empty:
                        df_chart = df_chart.sort_values(by=time_col_for_chart, ascending=True)

                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x=df_chart[time_col_for_chart], y=df_chart[selected_metric], ax=ax, palette='viridis') 

                        ax.set_title(f"Xu hướng {selected_metric} của {symbol} ({PERIOD_OPTIONS[period]})", fontsize=16)
                        ax.set_xlabel("Kỳ Báo Cáo", fontsize=12)
                        ax.set_ylabel(selected_metric, fontsize=12)
                        ax.ticklabel_format(style='plain', axis='y')
                        ax.grid(axis='y', linestyle='--', alpha=0.6)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning(f"Không có dữ liệu hợp lệ cho chỉ tiêu '{selected_metric}' để vẽ biểu đồ.")
                else:
                    st.warning("Không tìm thấy đủ dữ liệu (cột số hoặc cột thời gian) trong Báo cáo KQKD để trực quan hóa. Vui lòng kiểm tra cấu trúc dữ liệu.")

        # --- TAB PHÂN TÍCH AI TỔNG HỢP ---
        with tabs[5]: 
            st.subheader("Phân tích Chuyên sâu từ Gemini AI")
            st.markdown("Chức năng này sử dụng Bảng Thống kê (Tab 4) làm cơ sở để AI phân tích tình hình tài chính tổng thể của công ty.")
            
            if not api_key:
                st.error("Vui lòng nhập **GEMINI_API_KEY** vào Sidebar để kích hoạt chức năng này.")
            
            elif 'income_statement' not in stats_dfs or 'balance_sheet' not in stats_dfs:
                st.warning("Thiếu dữ liệu (KQKD hoặc Bảng Cân đối Kế toán) để tiến hành phân tích AI.")

            else:
                if st.button("🌟 Yêu cầu AI Phân tích Tổng hợp Báo cáo Tài chính"):
                    with st.spinner('Đang gửi dữ liệu thống kê và chờ Gemini phân tích...'):
                        
                        ai_result = get_ai_analysis(
                            stats_dfs['income_statement'], 
                            stats_dfs['balance_sheet'], 
                            symbol, 
                            PERIOD_OPTIONS[period], 
                            api_key
                        )
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                
else:
    st.info("Vui lòng nhập Mã Cổ Phiếu để bắt đầu.")
