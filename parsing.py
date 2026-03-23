import os
import urllib.request
import tarfile
import pandas as pd
from email import policy
from email.parser import BytesParser
from tqdm import tqdm


class EnronDownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_dataset(self):
        url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
        output_file = os.path.join(self.raw_dir, "enron_mail.tar.gz")

        if os.path.exists(output_file):
            print(f"Archive already exists: {output_file}")
            return output_file

        print(f"Downloading from {url}")
        print("This will take 10-20 minutes (423 MB)...")

        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"Downloaded to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

    def extract_dataset(self, archive_path):
        extract_dir = os.path.join(self.raw_dir, 'maildir')

        if os.path.exists(extract_dir):
            print(f"Dataset already extracted: {extract_dir}")
            return extract_dir

        print("Extracting archive...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=self.raw_dir)
            print(f"Extracted to {extract_dir}")
            return extract_dir
        except Exception as e:
            print(f"Error extracting: {e}")
            return None

    def parse_email_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)

            email_data = {
                'message_id': msg.get('Message-ID', ''),
                'date': msg.get('Date', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'subject': msg.get('Subject', ''),
                'cc': msg.get('Cc', ''),
                'bcc': msg.get('Bcc', ''),
                'body': '',
                'filepath': filepath
            }

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        try:
                            email_data['body'] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                        except:
                            pass
            else:
                try:
                    email_data['body'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                except:
                    email_data['body'] = str(msg.get_payload())

            return email_data

        except Exception as e:
            return None

    def parse_all_emails(self, maildir_path, max_emails=100000):
        print(f"Parsing up to {max_emails} emails...")

        emails = []
        count = 0

        for root, dirs, files in os.walk(maildir_path):
            for filename in files:
                if count >= max_emails:
                    break

                filepath = os.path.join(root, filename)

                try:
                    email_data = self.parse_email_file(filepath)

                    if email_data and email_data.get('body') and len(email_data['body']) > 10:
                        emails.append(email_data)
                        count += 1

                        if count % 1000 == 0:
                            print(f"Parsed {count} emails...")
                except:
                    pass

            if count >= max_emails:
                break

        print(f"Total parsed: {len(emails)} emails")
        return emails

    def save_to_csv(self, emails, filename='enron_emails_100k.csv'):
        output_path = os.path.join(self.processed_dir, filename)

        df = pd.DataFrame(emails)
        df.to_csv(output_path, index=False)

        print(f"\nSaved to {output_path}")
        print(f"Total emails: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")

        return output_path


def main():
    downloader = EnronDownloader()

    # Step 1: Download
    archive_path = downloader.download_dataset()
    if not archive_path:
        return

    # Step 2: Extract
    maildir_path = downloader.extract_dataset(archive_path)
    if not maildir_path:
        return

    # Step 3: Parse
    emails = downloader.parse_all_emails(maildir_path, max_emails=100000)

    if not emails:
        print("No emails parsed!")
        return

    # Step 4: Save
    output_path = downloader.save_to_csv(emails)

if __name__ == "__main__":
    main()