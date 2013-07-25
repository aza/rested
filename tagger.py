import sys
import json
import shelve
import md5

try:
	fileName = sys.argv[1]
except:
	print "Please specify a .json file name"
	sys.exit()

print fileName
f = open(fileName, 'r')

output = open('classified.txt', 'a')

seenComments = shelve.open('seen', writeback=True)

for line in f:
	data = json.loads(line)
	context = data['data'].get('context')
	comment = data['data'].get('comment')

	xid = data['xid']+data['timestamp']

	if seenComments.get(xid):
		print "Already tagged ", seenComments.get(xid), comment

	if seenComments.get(xid) == None and context == "sleep" and comment and "rest" in comment.lower():
		print comment
		input = raw_input("(w)ell rested, (p)oorly rested, enter for neither: " )
		if input == "w":
			output.write(input + '\t' + comment + '\t' + xid + '\n')
			print "Tagged as well rested\n"
		elif input == "p":
			print "Tagged as poorly rested\n"
			output.write(input + '\t' + comment + '\t' + xid + '\n')
		else:
			print ""
			input = "n"
		seenComments[xid] = input



