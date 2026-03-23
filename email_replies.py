import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm


def clean_email_body(text):
    if pd.isna(text):
        return ""

    text = str(text)

    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if line.strip().startswith('>'):
            continue
        if 'forwarded by' in line.lower():
            continue
        if 'original message' in line.lower():
            break
        if re.match(r'^(From|To|Subject|Date|Cc|Bcc):', line, re.IGNORECASE):
            continue

        clean_lines.append(line)

    cleaned = '\n'.join(clean_lines).strip()
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)

    return cleaned


def is_reply(subject):
    if pd.isna(subject):
        return False

    subject = str(subject).strip().lower()

    if subject.startswith('re:'):
        return True

    if subject.startswith('fwd:'):
        return False

    return False


def extract_original_subject(subject):
    if pd.isna(subject):
        return ""

    subject = str(subject).strip()

    while True:
        old_subject = subject
        subject = re.sub(r'^Re:\s*', '', subject, flags=re.IGNORECASE)
        subject = re.sub(r'^Fwd:\s*', '', subject, flags=re.IGNORECASE)
        subject = subject.strip()

        if subject == old_subject:
            break

    return subject


def find_email_pairs(df):
    print(f"Total emails: {len(df)}")

    subject_groups = defaultdict(list)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        subject = row.get('subject', '')
        if pd.isna(subject):
            continue

        original_subject = extract_original_subject(subject)

        if original_subject:
            subject_groups[original_subject].append({
                'index': idx,
                'subject': subject,
                'body': row.get('body', ''),
                'from': row.get('from', ''),
                'to': row.get('to', ''),
                'date': row.get('date', ''),
                'is_reply': is_reply(subject)
            })

    print(f"Found {len(subject_groups)} unique subjects")

    pairs = []

    for subject, emails in tqdm(subject_groups.items(), desc="Extracting pairs"):
        if len(emails) < 2:
            continue

        originals = [e for e in emails if not e['is_reply']]
        replies = [e for e in emails if e['is_reply']]

        if not originals or not replies:
            continue

        for original in originals[:2]:
            for reply in replies[:2]:

                original_body = clean_email_body(original['body'])
                reply_body = clean_email_body(reply['body'])

                if len(original_body.split()) < 3 or len(reply_body.split()) < 2:
                    continue

                if len(original_body.split()) > 300 or len(reply_body.split()) > 300:
                    continue

                pairs.append({
                    'original_subject': extract_original_subject(original['subject']),
                    'original_body': original_body,
                    'reply_body': reply_body,
                    'original_from': original['from'],
                    'reply_from': reply['from']
                })

    print(f"Found {len(pairs)} email-reply pairs")

    return pairs


def analyze_pairs(pairs):
    if not pairs:
        print("No pairs found")
        return

    original_lengths = [len(p['original_body'].split()) for p in pairs]
    reply_lengths = [len(p['reply_body'].split()) for p in pairs]

    print(f"\nOriginal emails - avg length: {sum(original_lengths) / len(original_lengths):.1f} words")
    print(f"Replies - avg length: {sum(reply_lengths) / len(reply_lengths):.1f} words")

    print("\nExamples:")
    for i, pair in enumerate(pairs[:3], 1):
        print(f"\nPair {i}:")
        print(f"Original: {pair['original_body'][:150]}...")
        print(f"Reply: {pair['reply_body'][:150]}...")


def save_pairs(pairs, output_path='./data/processed/email_reply_pairs.csv'):
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(pairs)} pairs to {output_path}")


def main():
    try:
        df = pd.read_csv('./data/processed/enron_emails_100k.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('./data/processed/enron_emails_classified.csv')
        except:
            print("Dataset not found")
            return

    print(f"Loaded {len(df)} emails")

    pairs = find_email_pairs(df)

    if not pairs:
        print("Could not find pairs")
        return

    analyze_pairs(pairs)
    save_pairs(pairs)


if __name__ == "__main__":
    main()