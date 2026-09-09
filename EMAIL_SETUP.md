# PDF email setup

The app sends the generated inspection PDF through the SMTP account configured
below. Create `.streamlit/secrets.toml` next to `app1.py` and add your sender
account values:

```toml
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USERNAME = "your-sender@gmail.com"
SMTP_PASSWORD = "your-16-character-gmail-app-password"
SMTP_FROM = "your-sender@gmail.com"
```

For Gmail, create an App Password in the Google account's Security settings;
your regular Gmail password will not work. Do not commit `secrets.toml` or
share its password.

Restart Streamlit after saving the file. In the app, enter the recipient's
email address, then select **Send PDF by email**. The recipient will receive
`Legal_Metrology_Certificate.pdf` as an attachment.
