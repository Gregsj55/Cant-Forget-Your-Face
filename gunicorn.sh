#!/bin/sh
gunicorn -k eventlet -w 1 apptest:app -b :5000
