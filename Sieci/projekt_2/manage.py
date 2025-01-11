#!/usr/bin/env python
import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sieci_public.settings')  # Replace 'sieci_public' with your project name
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Add a helpful message for collecting static files
    if len(sys.argv) > 1 and sys.argv[1] == 'collectstatic':
        print("Collecting static files...")
        print("Make sure your STATIC_ROOT is set correctly in settings.py.")

    # Execute the provided command
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
