constanta = -250;
numdocs = 50;
beep

for ii=251:250+numdocs
    movefile(['lengthForTerms',num2str(ii),'.txt'],['lengthForTerms',num2str(ii+constanta),'.txt']);
end


for ii=251:250+numdocs
    movefile(['termvars',num2str(ii),'.txt'],['termvars',num2str(ii+constanta),'.txt']);
end
for ii=251:250+numdocs
    movefile(['doc_id_strings',num2str(ii),'.txt'],['doc_id_strings',num2str(ii+constanta),'.txt']);
end