#!/bin/sh
# Genera atlas.html autocontenido (datos en línea) a partir de index.html + data.js + texts.js
cd "$(dirname "$0")"
awk '
/<script src="data.js"><\/script>/ { print "<script>"; while((getline l < "data.js")>0) print l; print "</script>"; next }
/<script src="texts.js"><\/script>/ { print "<script>"; while((getline l < "texts.js")>0) print l; print "</script>"; next }
{ print }' index.html > atlas.html
