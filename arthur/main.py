import streamlit as st

def main():
    lab_5 = st.Page('pages/lab_5.py', title='Лабораторная 5')
    lab_6 = st.Page('pages/lab_6.py', title='Лабораторная 6')
    lab_7 = st.Page('pages/lab_7.py', title='Лабораторная 7')
    pg = st.navigation([lab_5, lab_6, lab_7])
    pg.run()

if __name__ == '__main__':
    main()
