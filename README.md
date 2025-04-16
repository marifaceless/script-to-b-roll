# Script to B-Roll Finder

A Streamlit application that analyzes a script and finds relevant B-roll footage for each segment using the Pexels API.

## Features

- Takes a text script as input
- Breaks it down into segments of 3-5 seconds
- Extracts keywords from each segment
- Finds relevant B-roll footage from Pexels
- Displays results in a grid of thumbnails with download links

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/script-to-broll.git
   cd script-to-broll
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Paste your script into the text area
2. Adjust the segment duration slider if needed (3-5 seconds)
3. Click "Find B-Roll Footage"
4. Review the suggestions for each segment
5. Download the videos you like by clicking the download links

## Deployment to Streamlit Cloud

To deploy this app to Streamlit Cloud:

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Login with your GitHub account
4. Create a new app pointing to your repository
5. Add your Pexels API key as a secret:
   - Go to "Advanced settings" when deploying
   - Add secret: `PEXELS_API_KEY` with your API key value

## API Key

The app uses the Pexels API for finding stock videos. The API key is currently hardcoded for demonstration purposes, but for production use, you should:

1. Remove the hardcoded key from the code
2. Set it as an environment variable or Streamlit secret
3. Make sure to follow Pexels' attribution requirements when using their videos

## Attribution

When using videos from Pexels, be sure to credit the source. This is required by Pexels' license terms.

## License

MIT 