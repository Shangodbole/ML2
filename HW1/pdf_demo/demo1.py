import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('text2.pdf') as pdf:
    firstPage = plt.figure(figsize=(5, 5))
    firstPage.clf()
    txt = 'This is the title page'
    firstPage.text(0.0, 4.0, txt )
    firstPage.text(0.0, 2.5, "this is another line" )
    pdf.savefig()
    plt.close()