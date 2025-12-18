import pandas as pd

df_sui = pd.read_excel("aurora_sui_581-619.xlsx")
df_tang = pd.read_excel("aurora_tang_619-907.xlsx")
df_fdk = pd.read_excel("aurora_fivedynasties&10kingdoms_907-960.xlsx")
df_song = pd.read_excel("aurora_song_960-1279.xlsx")
df_yuan = pd.read_excel("aurora_yuan_1368-1644.xlsx")
df_ming = pd.read_excel("aurora_ming_1368-1644.xlsx")
df_qing = pd.read_excel("aurora_qing_1616-1949.xlsx")

df_sui["Source"] = "Sui"
df_tang["Source"] = "Tang"
df_fdk["Source"] = "FiveDynastiesTenKingdoms"
df_song["Source"] = "Song"
df_yuan["Source"] = "Yuan"
df_ming["Source"] = "Ming"
df_qing["Source"] = "Qing"

merged = pd.concat(
    [df_sui, df_tang, df_fdk, df_song, df_yuan, df_ming, df_qing], ignore_index=True
)
merged = merged.sort_values(by="Year").reset_index(drop=True)
merged.to_excel("Chinese_Aurora_Records.xlsx", index=False)
