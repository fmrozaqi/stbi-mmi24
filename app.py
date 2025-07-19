import streamlit as st
import requests

# Replace with your actual search API endpoint
API_URL = "http://localhost:8000/search"

st.title("Semantic Scholar Search Engine")

# Input query
query = st.text_input("Enter your search query:")

# Search button
if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query before searching.")
    else:
        # Call the API
        try:
            with st.spinner("Searching..."):
                response = requests.get(f"{API_URL}/{query}")
                response.raise_for_status()
                data = response.json()
                results = data.get("message", [])

                if results:
                    st.subheader("Search Results:")
                    for item in results:
                        st.markdown(f"##### [{item['text']}]({item['metadata']['url']})")
                        st.markdown(f"Year: {item['metadata']['year']}")
                        st.markdown(f"Authors: {item['metadata']['authors']}")
                        st.markdown(f"Abstract: {item['metadata']['abstract']}")
                        st.caption(f"Score: `{item["score"]}`")
                        st.markdown("---")
                else:
                    st.info("No results found.")

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
