How I like my scripts formatted:

User-controlled parameters as global variables at top of script
Comment dashed lines separating blocks of similarly functioning parameters

Plot formatting:

Tickmarks and axes boundary should be linewidth 2 and tickmarks should face inwardly
Capstyle on lines should be round

Always allow for user control (see above) of:
	linestyle
	linewidth
	marker
	linestyle color (including transparency separately)
	marker face color (including transparency separately)
	marker edge color (including transparency separately)
	axes size in mm and figure buffer around that in mm (by default axes should be 100 by 40)
	zorder of the various elements of the plot
	whether the grid is on or not
	whether there is a legend or not
	xlims and ylims

Always create a "for publication" option, which causes the plot to be saved as an svg with no title, no axes labels, and no tickmarklabels.
	
Ask any questions about requests that clear up what you need to do.

Do not save files (e.g. image files) to the same directory as the code. Save to a subfolder called media or else have the user input this directory.

