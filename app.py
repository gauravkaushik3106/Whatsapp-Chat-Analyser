import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("Whatsapp Chat Analyzer")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Choose WhatsApp chat (.txt)")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # WhatsApp Android exports are UTF-16
    try:
        data = bytes_data.decode("utf-16")
    except UnicodeDecodeError:
        data = bytes_data.decode("utf-8", errors="ignore")

    df = preprocessor.preprocess(data)

    # ✅ ONLY reject truly empty parses
    if df.shape[0] == 0:
        st.error("Chat file has no readable messages.")
        st.stop()

    # --------------------------------------------------
    # USER SELECTION
    # --------------------------------------------------
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        # ==================================================
        # TOP STATISTICS
        # ==================================================
        num_messages, words, num_media_messages, num_links, _ = helper.fetch_stats(
            selected_user, df
        )


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", words)
        col3.metric("Media", num_media_messages)
        col4.metric("Links", num_links)

        # ==================================================
        # MONTHLY TIMELINE
        # ==================================================
        if df.shape[0] >= 5:
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'])
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # ==================================================
        # DAILY TIMELINE
        # ==================================================
        if df.shape[0] >= 5:
            st.title("Daily Timeline")
            daily = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily['only_date'], daily['message'])
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # ==================================================
        # ACTIVITY MAP
        # ==================================================
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # ==================================================
        # WEEKLY HEATMAP
        # ==================================================
        if df.shape[0] >= 10:
            st.title("Weekly Activity Map")
            heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            sns.heatmap(heatmap, ax=ax)
            st.pyplot(fig)

        # ==================================================
        # WORDCLOUD
        # ==================================================
        st.title("Wordcloud")
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # ==================================================
        # EMOJI ANALYSIS
        # ==================================================
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    emoji_df[1].head(),
                    labels=emoji_df[0].head(),
                    autopct="%0.2f"
                )
                st.pyplot(fig)
        else:
            st.info("No emojis found in this chat.")

        # ==================================================
        # 🔥 EMOTION INTENSITY & EVENTS (SAFE + MEANINGFUL)
        # ==================================================
        st.title("Emotion Intensity & Conversation Events")

        # ✅ Guard: small chats cannot support time-series analytics
        if df.shape[0] < 10:
            st.warning(
                "Chat is too small for emotion intensity analysis. "
                "Add more messages to detect emotional patterns."
            )
        else:
            emotion_df = helper.compute_emotion_intensity(selected_user, df)

            # Remove NaN rows created by diff()
            emotion_df = emotion_df.dropna(subset=['emotion_intensity'])

            # Z-score normalization
            mean_intensity = emotion_df['emotion_intensity'].mean()
            std_intensity = emotion_df['emotion_intensity'].std()

            # Guard against zero variance
            if std_intensity == 0 or np.isnan(std_intensity):
                st.info("Not enough emotional variation to compute intensity.")
            else:
                emotion_df['z_intensity'] = (
                    (emotion_df['emotion_intensity'] - mean_intensity)
                    / std_intensity
                )

                emotion_df['z_smooth'] = emotion_df['z_intensity'].rolling(12).mean()
                events_df = helper.detect_emotional_events(emotion_df)

                # ----------------- GRAPH -----------------
                fig, ax = plt.subplots(figsize=(12, 4))

                ax.plot(
                    emotion_df['hour_block'],
                    emotion_df['z_smooth'],
                    color='black',
                    linewidth=1.5,
                    label='Smoothed Emotion Deviation'
                )

                ax.axhline(2, color='red', linestyle='--', alpha=0.6)
                ax.axhline(-2, color='green', linestyle='--', alpha=0.6)

                if not events_df.empty:
                    colors = events_df['event_type'].apply(
                        lambda x: 'red' if 'Negative' in x else 'green'
                    )
                    sizes = events_df['message_count'] * 8

                    ax.scatter(
                        events_df['hour_block'],
                        (events_df['emotion_intensity'] - mean_intensity) / std_intensity,
                        c=colors,
                        s=sizes,
                        alpha=0.7,
                        label='Detected Events'
                    )

                ax.set_title("Conversation Emotion Dynamics")
                ax.set_ylabel("Emotion Deviation (Z-score)")
                ax.set_xlabel("Time")
                ax.legend()
                plt.xticks(rotation=90)
                st.pyplot(fig)

                # ----------------- TABLE -----------------
                st.subheader("Detected Emotional Events")

                def highlight_events(row):
                    if "Negative" in row["event_type"]:
                        return ["background-color: #ffe6e6; color: black"] * len(row)
                    else:
                        return ["background-color: #e6ffe6; color: black"] * len(row)

                if not events_df.empty:
                    styled_df = events_df.style.apply(highlight_events, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.info("No statistically significant emotional events detected.")
