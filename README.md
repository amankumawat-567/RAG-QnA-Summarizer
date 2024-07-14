# RAG-QnA-Summarizer

This repository contains a Streamlit-based chatbot application that uses LangChain for document processing and question-answering capabilities. The chatbot can handle PDF and URL inputs to provide answers based on the contents of the documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install and run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/my-chatbot.git
   cd my-chatbot
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Install poppler:
   ```sh
   apt-get install -y poppler-utils
   ```

## Usage

To run the Streamlit app, execute the following command:
```sh
streamlit run app.py
```

### User Interface

- **PDFs**: Upload multiple PDF files via the sidebar to be processed by the chatbot.
- **URL**: Enter a URL in the sidebar to process the content from the webpage.

After processing the documents, you can ask questions in the chat input at the bottom of the page. The chatbot will provide responses based on the processed content.

## Features

- **Document Processing**: Upload PDFs or enter URLs to extract and process content.
- **Question Answering**: Ask questions about the content of the uploaded documents or URLs.
- **Document Summarization**: Summarize the content of the processed documents.

## Dependencies

This project relies on the following main dependencies:

- `langchain`
- `langchain_community`
- `langchain_cohere`
- `streamlit`
- `tempfile`
- `os`

For a complete list of dependencies, refer to the `requirements.txt` file.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the Apache License Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to contact us.