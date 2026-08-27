import pdfplumber
from backend.rag.pipeline import load_and_split_pdf


def reconstruct_diagram_text(pdf_path, page_num, row_tolerance=6,
                              cluster_gap=40, max_label_distance=60):
    
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        words = page.extract_words()

    rows = []
    for w in sorted(words, key=lambda w: w['top']):
        placed = False
        for row in rows:
            if abs(row[0]['top'] - w['top']) <= row_tolerance:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    rows.sort(key=lambda r: r[0]['top'])
    for row in rows:
        row.sort(key=lambda w: w['x0'])

    def cluster_row(row):
        clusters = [[row[0]]]
        for w in row[1:]:
            if w['x0'] - clusters[-1][-1]['x1'] > cluster_gap:
                clusters.append([w])
            else:
                clusters[-1].append(w)
        return clusters

    row_clusters = [cluster_row(r) for r in rows]
    row_tops = [r[0]['top'] for r in rows]

    def is_number_row(clusters):
        return all(len(c) == 1 and c[0]['text'].strip(').,').isdigit() for c in clusters)

    output_lines = []
    for i, clusters in enumerate(row_clusters):
        if is_number_row(clusters) and i > 0:
            label_clusters = []
            for j in range(i - 1, -1, -1):
                if row_tops[i] - row_tops[j] > max_label_distance:
                    break
                if not is_number_row(row_clusters[j]):
                    label_clusters.extend(row_clusters[j])
            for num_cluster in clusters:
                num_word = num_cluster[0]
                num_x = (num_word['x0'] + num_word['x1']) / 2

                def cluster_center(c):
                    return (c[0]['x0'] + c[-1]['x1']) / 2

                nearest = min(label_clusters, key=lambda c: abs(cluster_center(c) - num_x))
                label_text = ' '.join(w['text'] for w in nearest)
                output_lines.append(f"{label_text}: {num_word['text']}")
        else:
            for cluster in clusters:
                output_lines.append(' '.join(w['text'] for w in cluster))

    return '\n'.join(output_lines)


def load_pdf_with_layout_fix(pdf_path, pages_needing_fix=None):
  
    pages = load_and_split_pdf(pdf_path)  # baseline, preserves metadata
    if pages_needing_fix:
        for page_num in pages_needing_fix:
            pages[page_num].page_content = reconstruct_diagram_text(pdf_path, page_num)
    return pages