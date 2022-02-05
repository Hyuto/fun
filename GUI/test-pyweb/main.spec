# -*- mode: python ; coding: utf-8 -*-

import PyInstaller.config

PyInstaller.config.CONF['distpath'] = "./publish"
block_cipher = None

a = Analysis(['.\\main.py'],
             pathex=['.'],
             binaries=None,
             datas=[('.\\dist', 'dist')],
             hiddenimports=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='react-flask-pywebview-app',
          debug=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon='.\\public\\favicon.ico')
