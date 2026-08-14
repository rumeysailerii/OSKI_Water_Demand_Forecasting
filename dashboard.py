import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pazar Doygunluk ve Sarfiyat Analizi", layout="wide")

@st.cache_data
def abone_verisi_yukle(dosya_yolu):
    df = pd.read_excel(dosya_yolu, sheet_name='İlçe Mahalle Bazlı') 
    df.columns = df.columns.astype(str).str.strip()
    df['Abone Olan Sayısı'] = pd.to_numeric(df['Abone Olan Sayısı'], errors='coerce').fillna(0)
    df['ILCE_MAHALLE'] = df['ILCE'].astype(str) + " - " + df['MAHALLE_KOY'].astype(str)
    
    pivot_df = df.pivot_table(index='ILCE_MAHALLE', columns='YIL', values='Abone Olan Sayısı', aggfunc='sum').fillna(0)
    sayisal_yillar = [col for col in pivot_df.columns if isinstance(col, (int, float))]
    
    pivot_df['Toplam_Abone'] = pivot_df[sayisal_yillar].sum(axis=1)
    son_3_yil = [y for y in sayisal_yillar if y >= 2024]
    pivot_df['Son_3_Yil_Yeni_Kayit'] = pivot_df[son_3_yil].sum(axis=1) if son_3_yil else 0
        
    pivot_df['Buyume_Orani_%'] = np.where(
        pivot_df['Toplam_Abone'] > 0, 
        (pivot_df['Son_3_Yil_Yeni_Kayit'] / pivot_df['Toplam_Abone']) * 100, 
        0
    ).round(2)
    
    return pivot_df.reset_index()

@st.cache_data
def sarfiyat_verisi_yukle(dosya_yolu, sarfiyat_sutun_adi):
    tum_sekmeler = pd.read_excel(dosya_yolu, sheet_name=None)
    
    tum_veriler = []
    for sekme_adi, df in tum_sekmeler.items():
        df.columns = df.columns.astype(str).str.strip()
        
        if 'BOLGE' in df.columns and 'MAHALLE' in df.columns:
            df['ILCE_MAHALLE'] = df['BOLGE'].astype(str) + " - " + df['MAHALLE'].astype(str)
            tum_veriler.append(df[['ILCE_MAHALLE', sarfiyat_sutun_adi]])
            
    birlesik_df = pd.concat(tum_veriler, ignore_index=True)
    birlesik_df[sarfiyat_sutun_adi] = pd.to_numeric(birlesik_df[sarfiyat_sutun_adi], errors='coerce').fillna(0)
    
    mahalle_sarfiyat = birlesik_df.groupby('ILCE_MAHALLE')[sarfiyat_sutun_adi].sum().reset_index()
    mahalle_sarfiyat.rename(columns={sarfiyat_sutun_adi: 'Toplam_Sarfiyat'}, inplace=True)
    
    return mahalle_sarfiyat

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3067/3067406.png", width=100)
st.sidebar.title("Kontrol Paneli")

abone_dosyasi = st.sidebar.text_input("1. Abone Excel Yolu:", r"C:\Users\rumey\Downloads\RÜMEYSARAPOR3.xlsx")
sarfiyat_dosyasi = st.sidebar.text_input("2. Sarfiyat Excel Yolu:", r"C:\Users\rumey\Downloads\Rümeysa rapor2.xlsx")
st.sidebar.markdown("---")
ariza_excel_yolu = st.sidebar.text_input("3. Arıza Excel Yolu:", r"C:\Users\rumey\Downloads\Rümeysa-4  İlçe Bazlı 2020-2025 Arıza Kayıtları.xlsx")
sarfiyat_sutunu = st.sidebar.text_input("Sarfiyat Sütun Adı:", "TOPLAM_SARFIYAT") 

st.sidebar.markdown("---")
secilen_oran = st.sidebar.slider("Doygunluk Sınırı (%)", 1.0, 20.0, 5.0, 0.5)
kume_sayisi = st.sidebar.slider("K-Means Küme Sayısı", 2, 6, 4, 1)

st.title("💧 Entegre Abone ve Sarfiyat Strateji Dashboard'u")

try:
    abone_df = abone_verisi_yukle(abone_dosyasi).copy()
    sarfiyat_df = sarfiyat_verisi_yukle(sarfiyat_dosyasi, sarfiyat_sutunu).copy()
    
    veri = pd.merge(abone_df, sarfiyat_df, on='ILCE_MAHALLE', how='left').fillna(0)
    
    def dinamik_doygunluk(row):
        if row['Toplam_Abone'] < 30: return "Düşük Hacimli"
        elif row['Buyume_Orani_%'] <= secilen_oran: return "Doygunluğa Ulaştı"
        else: return "Aktif Büyüyor"
        
    veri['Dinamik_Durum'] = veri.apply(dinamik_doygunluk, axis=1)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Oransal Dağılım", 
        "Yapay Zeka", 
        "Filtreli Tablo", 
        "Büyüme-Sarfiyat Matrisi", 
        "Altyapı ve Arıza Analizi"
    ])

    with tab1:
        st.subheader(f"Yüzde {secilen_oran} Büyüme Barajına Göre Doygunluk")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=veri, x='Toplam_Abone', y='Buyume_Orani_%', hue='Dinamik_Durum',
            palette={'Düşük Hacimli': 'gray', 'Doygunluğa Ulaştı': 'red', 'Aktif Büyüyor': 'green'},
            s=80, alpha=0.8, edgecolor='black', ax=ax
        )
        ax.axhline(y=secilen_oran, color='red', linestyle='--', alpha=0.6)
        ax.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig)

    with tab2:
        st.subheader(f"K-Means ile {kume_sayisi} Farklı Gruba Ayırma")
        ml_verisi = veri[['Toplam_Abone', 'Son_3_Yil_Yeni_Kayit']]
        scaler = StandardScaler()
        olcekli = scaler.fit_transform(ml_verisi)
        kmeans = KMeans(n_clusters=kume_sayisi, random_state=42, n_init=10)
        veri['ML_Kumesi'] = kmeans.fit_predict(olcekli)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=veri, x='Toplam_Abone', y='Son_3_Yil_Yeni_Kayit', hue='ML_Kumesi',
            palette="Set1", s=100, alpha=0.8, edgecolor='black', ax=ax2
        )
        ax2.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Tüm Mahallelerin Güncel Verisi")
        secilen_durum = st.radio("Filtrele:", ["Tümü", "Doygunluğa Ulaştı", "Aktif Büyüyor", "Düşük Hacimli"], horizontal=True)
        if secilen_durum == "Tümü": 
            gosterilecek_veri = veri
        else: 
            gosterilecek_veri = veri[veri['Dinamik_Durum'] == secilen_durum]
            
        st.dataframe(gosterilecek_veri[['ILCE_MAHALLE', 'Toplam_Abone', 'Son_3_Yil_Yeni_Kayit', 'Buyume_Orani_%', 'Toplam_Sarfiyat', 'Dinamik_Durum']].sort_values(by='Toplam_Abone', ascending=False))

    with tab4:
        st.subheader("Büyüme Hızı vs Toplam Sarfiyat Matrisi")
        st.markdown("**Balonların üzerine fareyle gelerek mahalle detaylarını inceleyebilirsiniz.**")
        
        bubble_veri = veri[veri['Toplam_Abone'] >= 30].copy()
        
        fig4 = px.scatter(
            bubble_veri, 
            x='Toplam_Abone', 
            y='Buyume_Orani_%', 
            size='Toplam_Sarfiyat',
            color='Dinamik_Durum',
            hover_name='ILCE_MAHALLE', 
            hover_data={'Toplam_Abone': True, 'Buyume_Orani_%': True, 'Toplam_Sarfiyat': True},
            color_discrete_map={'Doygunluğa Ulaştı': 'red', 'Aktif Büyüyor': 'green', 'Düşük Hacimli': 'gray'},
            size_max=60, 
            opacity=0.7
        )
        
        fig4.add_hline(y=secilen_oran, line_dash="dash", line_color="red", opacity=0.8)
    
        fig4.update_layout(
            title="Stratejik Değer Haritası: Üzerine Gelip Keşfedin",
            xaxis_title="Tüm Yıllardaki Toplam Abone Sayısı (Hacim)",
            yaxis_title="Büyüme Oranı (%)",
            plot_bgcolor="white",
            hovermode="closest",
            height=650 
        )
        fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        st.subheader("🛠️ Altyapı ve Arıza Analizi")
        
        try:
            df_ariza = pd.read_excel(ariza_excel_yolu, usecols=[0, 1, 2, 3, 4])
            df_ariza.columns = ['Yıl', 'İlçe', 'İhbar Türü', 'Gelen Arıza', 'Sonuçlanan Arıza']
            
            st.markdown("#### 📌 İlçelere Göre Toplam Arıza Yoğunluğu")
            ilce_ariza = df_ariza.groupby('İlçe')['Gelen Arıza'].sum().reset_index()
            ilce_ariza = ilce_ariza.sort_values(by='Gelen Arıza', ascending=False)
            
            fig_ilce = px.bar(
                ilce_ariza, 
                x='İlçe', 
                y='Gelen Arıza', 
                text='Gelen Arıza',
                color='Gelen Arıza',
                color_continuous_scale='Reds',
                labels={'İlçe': 'İlçe / Mahalle', 'Gelen Arıza': 'Toplam Arıza Kaydı'}
            )
            fig_ilce.update_traces(textposition='outside')
            st.plotly_chart(fig_ilce, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚠️ Arıza Türü Dağılımı")
                turu_ariza = df_ariza.groupby('İhbar Türü')['Gelen Arıza'].sum().reset_index()
                
                fig_tur = px.pie(
                    turu_ariza, 
                    values='Gelen Arıza', 
                    names='İhbar Türü', 
                    hole=0.4, 
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_tur.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_tur, use_container_width=True)
                
            with col2:
                st.markdown("#### 📈 Yıllara Göre Arıza Trendi")
                yil_ariza = df_ariza.groupby('Yıl')['Gelen Arıza'].sum().reset_index()
                
                fig_yil = px.line(
                    yil_ariza, 
                    x='Yıl', 
                    y='Gelen Arıza', 
                    markers=True,
                    labels={'Yıl': 'Yıl', 'Gelen Arıza': 'Toplam Arıza Kaydı'}
                )
                fig_yil.update_traces(line=dict(color='orange', width=4), marker=dict(size=10, color='red'))
                fig_yil.update_xaxes(dtick=1) 
                st.plotly_chart(fig_yil, use_container_width=True)
                
            # --- KIYASLAMA BÖLÜMÜ: ABONE SAYISI VS ARIZA KAYDI ---
            st.markdown("---")
            st.markdown("### 🚨 Altyapı Yatırım Karar Matrisi (Abone Büyümesi vs. Arıza)")
            
            abone_df_ilce = abone_df.copy()
            abone_df_ilce['İlçe_Temiz'] = abone_df_ilce['ILCE_MAHALLE'].apply(lambda x: str(x).split(' - ')[0].strip().upper())
            ilce_abone_ozet = abone_df_ilce.groupby('İlçe_Temiz')[['Toplam_Abone', 'Son_3_Yil_Yeni_Kayit']].sum().reset_index()
            
            df_ariza_ilce = df_ariza.copy()
            df_ariza_ilce['İlçe_Temiz'] = df_ariza_ilce['İlçe'].astype(str).str.strip().str.upper()
            ilce_ariza_ozet = df_ariza_ilce.groupby('İlçe_Temiz')['Gelen Arıza'].sum().reset_index()
            
            yatirim_df = pd.merge(ilce_abone_ozet, ilce_ariza_ozet, on='İlçe_Temiz', how='inner')
            
            yatirim_df['Stres_Skoru'] = np.where(
                yatirim_df['Toplam_Abone'] > 0,
                (yatirim_df['Gelen Arıza'] / yatirim_df['Toplam_Abone']) * 100,
                0
            ).round(2)
            
            fig_yatirim = px.scatter(
                yatirim_df,
                x='Son_3_Yil_Yeni_Kayit',
                y='Stres_Skoru',
                size='Gelen Arıza',
                color='Stres_Skoru',
                hover_name='İlçe_Temiz',
                text='İlçe_Temiz',
                color_continuous_scale='Turbo',
                labels={
                    'Son_3_Yil_Yeni_Kayit': 'Son 3 Yıldaki Yeni Abone Sayısı (Büyüme Hızı)',
                    'Stres_Skoru': 'Altyapı Stres Skoru (100 Abonede Arıza Sayısı)'
                }
            )
            fig_yatirim.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig_yatirim.update_layout(height=600, plot_bgcolor="white", title="Stratejik Yatırım Haritası")
            fig_yatirim.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig_yatirim.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig_yatirim, use_container_width=True)
            
            
            st.markdown("---")
            st.markdown("### 🔮 Makine Öğrenmesi & Özellik Mühendisliği (Büyük 3 İlçe Odaklı)")
            st.info("""
            **Yönetici Özeti:** OSKİ'nin en büyük lojistik ve altyapı yükünü çeken "Büyük Üçlü" (Altınordu, Fatsa, Ünye) ilçeleri için Lineer, Random Forest ve KNN modelleri karşılaştırılmıştır.
            """)
            
            df_tanker = df_ariza[df_ariza['İhbar Türü'].str.contains("TANKER", case=False, na=False)].copy()
            
            if not df_tanker.empty:
                tanker_grup = df_tanker.groupby(['İlçe', 'Yıl'])['Gelen Arıza'].sum().reset_index()
                tanker_grup['İlçe_Temiz'] = tanker_grup['İlçe'].astype(str).str.strip().str.upper()
                tanker_gelismis = pd.merge(tanker_grup, ilce_abone_ozet, on='İlçe_Temiz', how='left').fillna(0)
                
                X_gelismis = tanker_gelismis[['Yıl', 'İlçe', 'Toplam_Abone', 'Son_3_Yil_Yeni_Kayit']]
                X_gelismis_encoded = pd.get_dummies(X_gelismis, columns=['İlçe'])
                y = tanker_grup['Gelen Arıza']
                
                X_train, X_test, y_train, y_test = train_test_split(X_gelismis_encoded, y, test_size=0.2, random_state=42)
                kf = KFold(n_splits=min(5, len(X_gelismis_encoded)), shuffle=True, random_state=42)
                
                
                lr = LinearRegression().fit(X_train, y_train)
                rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
                knn = KNeighborsRegressor(n_neighbors=min(3, len(X_train))).fit(X_train, y_train)
                
                
                metrik_data = []
                for model, name in [(lr, 'Lineer'), (rf, 'RF'), (knn, 'KNN')]:
                    pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    cv_r2 = cross_val_score(model, X_gelismis_encoded, y, cv=kf, scoring='r2').mean()
                    metrik_data.append([name, mae, r2, cv_r2])
                
                metrik_df = pd.DataFrame(metrik_data, columns=['Model', 'MAE', 'Normal R2', 'CV R2']).set_index('Model')
                
                st.dataframe(metrik_df.style.format("{:.2f}"))
                
                # 2026 Tahmin
                df_2026 = pd.merge(pd.DataFrame({'İlçe': tanker_grup['İlçe'].unique()}), ilce_abone_ozet, left_on='İlçe', right_on='İlçe_Temiz', how='left').fillna(0)
                df_2026['Yıl'] = 2026
                X_2026 = pd.get_dummies(df_2026[['Yıl', 'İlçe', 'Toplam_Abone', 'Son_3_Yil_Yeni_Kayit']], columns=['İlçe'])
                for col in X_gelismis_encoded.columns:
                    if col not in X_2026.columns: X_2026[col] = 0
                X_2026 = X_2026[X_gelismis_encoded.columns]
                
                df_2026['LR'] = lr.predict(X_2026).round()
                df_2026['RF'] = rf.predict(X_2026).round()
                df_2026['KNN'] = knn.predict(X_2026).round()
                
                df_final = df_2026[df_2026['İlçe'].isin(['ALTINORDU', 'FATSA', 'ÜNYE'])]
                
                fig_f = go.Figure([
                    go.Bar(name='Lineer', x=df_final['İlçe'], y=df_final['LR']),
                    go.Bar(name='RF', x=df_final['İlçe'], y=df_final['RF']),
                    go.Bar(name='KNN', x=df_final['İlçe'], y=df_final['KNN'])
                ])
                st.plotly_chart(fig_f, use_container_width=True)
                st.dataframe(df_final[['İlçe', 'LR', 'RF', 'KNN']])

        except Exception as e:
            st.warning("Veri yüklenemedi.")
except Exception as e:
    st.error(f"Hata: {e}")
