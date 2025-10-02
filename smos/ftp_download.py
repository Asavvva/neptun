import ftplib
import os
import re
import socket
import ssl


class ImplicitFTP_TLS(ftplib.FTP_TLS):
    """FTP_TLS subclass that automatically wraps sockets in SSL to support implicit FTPS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sock = None

    @property
    def sock(self):
        """Return the socket."""
        return self._sock

    @sock.setter
    def sock(self, value):
        """When modifying the socket, ensure that it is ssl wrapped."""
        if value is not None and not isinstance(value, ssl.SSLSocket):
            value = self.context.wrap_socket(value)
        self._sock = value
        
        
def ftp_download(server, username, password, remote_dir, years, months, days):
    # Создаем объект FTP_TLS вместо FTP
    print(f"Start connecting to {server}...")
    ftp = ImplicitFTP_TLS()
    
    # Подключаемся к серверу на порту 990, используем для неявного FTPS
    ftp.connect(server, 990)
    
    # Входим в систему с использованием предоставленных учетных данных
    ftp.login(user=username, passwd=password)

    # Устанавливаем режим передачи данных в защищенный режим
    ftp.prot_p()

    for year in years:
        try:
            os.makedirs(f'/mnt/hippocamp/DATA/sattelite/SMOS/L2OS/MIR_OSUDP2_nc/{year}')
        except:
            pass
        for month in months:
            try:
                os.makedirs(f'/mnt/hippocamp/DATA/sattelite/SMOS/L2OS/MIR_OSUDP2_nc/{year}/{month:02}')
            except:
                pass
            for day in days:
                try:
                    os.makedirs(f'/mnt/hippocamp/DATA/sattelite/SMOS/L2OS/MIR_OSUDP2_nc/{year}/{month:02}/{day:02}')
                except:
                    pass

                remote_path = f"{remote_dir}/{year}/{month:02}/{day:02}/"
                try:
                    # Получаем список файлов в удаленной директории
                    files = ftp.nlst(remote_path)
                    
                    # Фильтруем файлы по расширению .nc
                    nc_files = [f for f in files if re.search(r'\.nc$', f)]
                    # print('Надо подкачаться')
                    for file in nc_files:
                        remote_filename = os.path.join(remote_path, file)
                        # list_files_in_directory(ftp, remote_filename)
                        local_filename = os.path.join(f'/mnt/hippocamp/DATA/sattelite/SMOS/L2OS/MIR_OSUDP2_nc/{year}/{month:02}/{day:02}',
                                                      os.path.basename(file))
                        if not os.path.exists(local_filename):
                            with open(local_filename, 'wb') as f:
                                ftp.retrbinary(f'RETR {remote_filename}', f.write)
                                print(f'Скачан: {local_filename}')
                except ftplib.error_perm as e:
                    print(f"Ошибка доступа к {remote_path}: {e}")

    # Закрываем FTP-соединение
    ftp.quit()


def list_files_in_directory(ftp, remote_dir):
    try:
        # Получаем список файлов и каталогов в указанном каталоге
        files = ftp.nlst(remote_dir)
        print(f"Список файлов в каталоге {remote_dir}:")
        for file in files:
            print(file)
    except ftplib.error_perm as e:
        print(f"Ошибка доступа к каталогу {remote_dir}: {e}")
        
        
def check_connection(server):
    try:
        print(f"Resolving {server}...")
        ip_address = socket.gethostbyname(server)
        print(f"Resolved IP: {ip_address}")
        
        print(f"Connecting to {server}...")
        print("Connection successful!")
    except socket.gaierror as e:
        print(f"Name resolution error: {e}")
    except socket.error as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    # SERVER = 'smos-diss.eo.esa.int'
    SERVER = '131.176.196.6'
    USERNAME = 'avellinnaa@gmail.com'  # Укажите ваше имя пользователя
    PASSWORD = 'Avelina2002'   # Укажите ваш пароль
    REMOTE_DIR = '/SMOS/L2OS/MIR_OSUDP2_nc'
    
   

    # Задаем диапазоны годов и месяцев
    years = range(2011, 2012)  # 2014 - 2024
    months = range(6, 12)       # 01 - 06
    days = range(1, 32)
    check_connection(SERVER)
    ftp_download(SERVER, USERNAME, PASSWORD, REMOTE_DIR, years, months, days)


#ftp:avellinnaa@gmail.com:Avelina2002@smos-diss.eo.esa.int

