(TeX-add-style-hook
 "Project"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8") ("geometry" "top=2.5cm" "left=3cm" "right=3cm" "bottom=4.0cm") ("xcolor" "table")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "babel"
    "inputenc"
    "amsmath"
    "amssymb"
    "parskip"
    "graphicx"
    "geometry"
    "xcolor")
   (TeX-add-symbols
    "tablespace"
    "Tstrut"
    "tstrut"
    "Bstrut"))
 :latex)

