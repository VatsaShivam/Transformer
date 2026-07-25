"""
A small Flask web app around the story QA system built in qa_retrieval.py.

Type a question in the browser, get back the most relevant sentence from
the story, grounded (every answer is a real sentence from the source
text -- it can't invent facts that aren't there) via TF-IDF + cosine
similarity retrieval.

Run with:
    python3 app.py
Then open:
    http://127.0.0.1:5000

Optionally load your own text instead of the built-in story:
    python3 app.py --file mytext.txt
"""

import argparse

from flask import Flask, render_template_string, request

from qa_retrieval import CORPUS, split_sentences, TfidfIndex

app = Flask(__name__)

# Populated in main() once we know which text to index (built-in story or
# a user-supplied file), so the index is built once at startup rather
# than on every request.
qa_index: TfidfIndex = None
source_label = "the lighthouse story"

# In-memory conversation history for this process (no database -- simple
# demo state, resets when the server restarts).
history = []

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Story QA</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
           max-width: 720px; margin: 40px auto; padding: 0 20px; color: #222; }
    h1 { font-size: 1.4rem; }
    .subtitle { color: #666; margin-top: -8px; }
    form { display: flex; gap: 8px; margin: 24px 0; }
    input[type=text] { flex: 1; padding: 10px 12px; font-size: 1rem;
                        border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 10px 18px; font-size: 1rem; border: none;
              border-radius: 6px; background: #2563eb; color: white; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .entry { border: 1px solid #e5e5e5; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }
    .question { font-weight: 600; }
    .answer { margin-top: 6px; }
    .score { color: #888; font-size: 0.85rem; margin-top: 4px; }
    .not-found { color: #b91c1c; }
    .empty { color: #888; font-style: italic; }
  </style>
</head>
<body>
  <h1>Story QA</h1>
  <p class="subtitle">Answers are retrieved sentences from {{ source_label }} -- not generated, so they can't be made up.</p>

  <form method="POST" action="/">
    <input type="text" name="question" placeholder="Ask a question about the text..." autofocus>
    <button type="submit">Ask</button>
  </form>

  {% if not history %}
    <p class="empty">No questions asked yet.</p>
  {% endif %}

  {% for item in history|reverse %}
    <div class="entry">
      <div class="question">Q: {{ item.question }}</div>
      {% if item.answer %}
        <div class="answer">A: {{ item.answer }}</div>
        <div class="score">similarity score: {{ "%.3f"|format(item.score) }}</div>
      {% else %}
        <div class="answer not-found">Not found in the text (best match score {{ "%.3f"|format(item.score) }})</div>
      {% endif %}
    </div>
  {% endfor %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer, score = qa_index.answer(question)
            history.append({"question": question, "answer": answer, "score": score})
    return render_template_string(PAGE_TEMPLATE, history=history, source_label=source_label)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """JSON endpoint: POST {"question": "..."} -> {"answer": ..., "score": ...}"""
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return {"error": "question is required"}, 400
    answer, score = qa_index.answer(question)
    return {"question": question, "answer": answer, "score": score}


def build_index(text: str) -> TfidfIndex:
    sentences = split_sentences(text)
    return TfidfIndex(sentences)


def main():
    global qa_index, source_label

    parser = argparse.ArgumentParser(description="Story QA web app")
    parser.add_argument("--file", type=str, default=None,
                         help="Path to a .txt file to index instead of the built-in story")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        source_label = f"'{args.file}'"
    else:
        text = CORPUS
        source_label = "the lighthouse story"

    qa_index = build_index(text)
    print(f"Indexed {len(qa_index.sentences)} sentences from {source_label}.")
    print(f"Open http://127.0.0.1:{args.port} in your browser.")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
