import ftplib
import os
import re

def ftp_download(server, username, password, remote_dir, years, months):
    # Подключаемся к FTP-серверу
    ftp = ftplib.FTP(server)
    ftp.login(user=username, passwd=password)

    # # Создаем локальную директорию для загрузки файлов
    # os.makedirs('downloaded_files', exist_ok=True)

    for year in years:
        try:
            os.makedirs(f'/mnt/hippocamp/DATA/sattelite/SMAP_V6.0/L2C/{year}')
        except:
            pass
        for month in months:
            try:
                os.makedirs(f'/mnt/hippocamp/DATA/sattelite/SMAP_V6.0/L2C/{year}/{month:02}')
            except:
                pass

            remote_path = f"{remote_dir}/{year}/{month:02}/"
            try:
                # Получаем список файлов в удаленной директории
                files = ftp.nlst(remote_path)
                
                # Фильтруем файлы по расширению .nc
                nc_files = [f for f in files if re.search(r'\.nc$', f)]

                for file in nc_files:
                    local_filename = os.path.join(f'/mnt/hippocamp/DATA/sattelite/SMAP_V6.0/L2C/{year}/{month:02}',
                                                  os.path.basename(file))
                    if not os.path.exists(local_filename):
                        with open(local_filename, 'wb') as f:
                            ftp.retrbinary(f'RETR {file}', f.write)
                            print(f'Скачан: {local_filename}')
            except ftplib.error_perm as e:
                print(f"Ошибка доступа к {remote_path}: {e}")

    # Закрываем FTP-соединение
    ftp.quit()

if __name__ == "__main__":
    SERVER = 'ftp.remss.com'
    USERNAME = 'alexanderssavin@gmail.com'  # Укажите ваше имя пользователя
    PASSWORD = 'alexanderssavin@gmail.com'   # Укажите ваш пароль
    REMOTE_DIR = '/smap/SSS/V06.0/FINAL/L2C'

    # Задаем диапазоны годов и месяцев
    years = range(2015, 2016)  # 2014 - 2024
    months = range(6, 12)       # 01 - 06

    ftp_download(SERVER, USERNAME, PASSWORD, REMOTE_DIR, years, months)
