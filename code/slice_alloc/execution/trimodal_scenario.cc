// Task-driven IP / restricted EID-RLOC / native NFD coexistence. Independent of
// protocol_probe.cc. Infrastructure WiFi association is driven by radio loss.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/ndnSIM-module.h"
#include "ns3/ndnSIM/apps/ndn-app.hpp"
#include "ns3/ndnSIM/model/ndn-app-link-service.hpp"
#include "ns3/ndnSIM/model/ndn-block-header.hpp"
#include "ns3/ndnSIM/NFD/daemon/face/generic-link-service.hpp"
#include "ns3/ndnSIM/NFD/daemon/table/cs-policy.hpp"
#include "ns3/wifi-phy-state-helper.h"
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <deque>
using namespace ns3;
namespace {
NodeContainer nodes;
std::ofstream events;
uint32_t uavs=4; double ttl=.6, stopAt=25., speed=8., startX=0.;
std::map<uint32_t,Ptr<NetDevice>> radios;
std::map<std::pair<uint32_t,uint32_t>,std::shared_ptr<nfd::Face>> faces;
std::map<std::pair<uint32_t,uint32_t>,std::pair<Ptr<NetDevice>,Ipv4Address>> hops;
std::map<uint32_t,Ptr<Socket>> sockets;
std::map<uint32_t,uint32_t> attached,versions;
std::map<uint32_t,Ipv4Address> addresses;
std::map<uint32_t,Ipv4Address> edgeAddress;
double Now(){return Simulator::Now().GetSeconds();}
void Log(uint32_t node,std::string event,uint32_t sid,uint32_t seq,uint64_t bytes,double detail=0){
 events<<std::setprecision(12)<<Now()<<","<<node<<","<<event<<","<<sid<<","<<seq<<","<<bytes<<","<<detail<<"\n";
}
struct Task {uint32_t id,tenant,uav,edge,server,cache; std::string mode;double start,end,period,deadline,drain;std::string prefix;};
std::map<uint32_t,Task> tasks;
struct Mapping{uint32_t version=0,locator=0;double until=0;};
std::map<std::pair<uint32_t,uint32_t>,Mapping> mappings,caches;
struct Pending{double born;uint32_t size;EventId retry;};
std::map<uint32_t,uint32_t> registrationAck;
std::map<std::pair<uint32_t,uint32_t>,Pending> pending;
std::map<std::pair<uint32_t,uint32_t>,double> born;
std::set<std::pair<uint32_t,uint32_t>> delivered;
std::map<uint32_t,uint32_t> queryTxn;
std::map<uint32_t,double> queryUntil;
// Tenant and endpoint are checked against immutable per-instance task metadata.
// 25-byte restricted experimental header, not an interoperable LISP encoding.
class Message:public Header{
public:uint8_t type=0;uint32_t sid=0,eid=0,seq=0,version=0,locator=0,txn=0;
 static TypeId GetTypeId(){static TypeId t=TypeId("ns3::TrimodalMessage").SetParent<Header>().AddConstructor<Message>();return t;}
 TypeId GetInstanceTypeId() const override{return GetTypeId();}
 uint32_t GetSerializedSize()const override{return type==5?9:25;}
 void Serialize(Buffer::Iterator b)const override{b.WriteU8(type);b.WriteHtonU32(sid);if(type==5){b.WriteHtonU32(seq);return;}for(uint32_t v:{eid,seq,version,locator,txn})b.WriteHtonU32(v);}
 uint32_t Deserialize(Buffer::Iterator b)override{type=b.ReadU8();sid=b.ReadNtohU32();if(type==5){seq=b.ReadNtohU32();return 9;}eid=b.ReadNtohU32();seq=b.ReadNtohU32();version=b.ReadNtohU32();locator=b.ReadNtohU32();txn=b.ReadNtohU32();return 25;}
 void Print(std::ostream&o)const override{o<<unsigned(type)<<":"<<sid<<":"<<eid;}
};
NS_OBJECT_ENSURE_REGISTERED(Message);
std::map<std::pair<uint32_t,uint8_t>,uint32_t> sentBudget;
void Send(uint32_t from,Ipv4Address to,Message h,uint32_t payload=0){
 const auto&t=tasks.at(h.sid);uint32_t count=0;for(double at=t.start+.05;at<t.end-1e-8;at+=t.period)++count;
 uint32_t limit=(h.type==1||h.type==6||h.type==9)?24:count*(uint32_t(std::ceil(t.deadline/.2))+1);
 if(h.type==7)limit=count;
 if(++sentBudget[{h.sid,h.type}]>limit){Log(from,"budget_drop",h.sid,h.seq,payload,h.type);return;}

 auto p=Create<Packet>(payload);
 if(h.type==4){
 Message inner;inner.type=5;inner.sid=h.sid;inner.seq=h.seq;p->AddHeader(inner);
 UdpHeader udp;udp.SetSourcePort(9000);udp.SetDestinationPort(9001);p->AddHeader(udp);
 Ipv4Header ip;ip.SetSource(Ipv4Address("10.238.255.1"));ip.SetDestination(Ipv4Address(0x0aee0000+h.eid));ip.SetProtocol(17);ip.SetTtl(64);ip.SetPayloadSize(p->GetSize());p->AddHeader(ip);
 }
 p->AddHeader(h);sockets.at(from)->SendTo(p,0,InetSocketAddress(to,4341));
 Log(from,"udp_send",h.sid,h.seq,p->GetSize()+28,h.type); // actual IPv4/UDP layer
}
void State(){
 for(uint32_t n:{1,2}){
 uint32_t a=0,c=0;for(auto&v:mappings)if(v.first.first==n)++a;for(auto&v:caches)if(v.first.first==n&&v.second.until>Now())++c;
 Log(n,"map_state",0,0,a,c);
 }
}
void Try(uint32_t sid,uint32_t seq);
void Register(uint32_t sid,uint32_t version,uint32_t attempt){
 const auto&t=tasks.at(sid);if(Now()>t.drain||attached[t.uav]==0||version!=versions[t.uav]||attempt>=12)return;
 if(registrationAck[sid]>=version)return;
 Message h;h.sid=sid;h.eid=t.uav;h.version=version;h.locator=addresses[t.uav].Get();h.type=t.mode=="id"?1:6;
 Send(t.uav,edgeAddress.at(t.server),h);Log(t.uav,"register",sid,version,53,h.locator);
 Simulator::Schedule(Seconds(.2),&Register,sid,version,attempt+1);
}
void Receive(Ptr<Socket> socket){Address from;while(auto p=socket->RecvFrom(from)){
 if(p->GetSize()<9){continue;}Message h;p->RemoveHeader(h);uint32_t n=socket->GetNode()->GetId();
 auto f=tasks.find(h.sid);if(f==tasks.end()||Now()>f->second.drain){Log(n,"late_drop",h.sid,h.seq,p->GetSize());continue;}
 const auto&t=f->second;if(h.type==5&&t.mode=="ip")h.eid=t.uav;
 if(h.eid!=t.uav){Log(n,"namespace_drop",h.sid,h.seq,0);continue;}
 Log(n,"udp_receive",h.sid,h.seq,p->GetSize()+h.GetSerializedSize()+28,h.type);
 if((h.type==1||h.type==6)&&n==t.server){
 auto& m=mappings[{n,h.sid}];if(h.version<m.version){Log(n,"stale_register",h.sid,h.seq,0);continue;}
 m={h.version,h.locator,t.drain};Log(n,"mapping_update",h.sid,h.version,0,h.locator);h.type=9;Send(n,InetSocketAddress::ConvertFrom(from).GetIpv4(),h);State();
 }else if(h.type==9&&n==t.uav){registrationAck[h.sid]=std::max(registrationAck[h.sid],h.version);
 }else if(h.type==2&&n==t.server){
 auto m=mappings.find({n,h.sid});if(m==mappings.end())continue;
 h.type=3;h.version=m->second.version;h.locator=m->second.locator;Send(n,InetSocketAddress::ConvertFrom(from).GetIpv4(),h);
 }else if(h.type==3&&n==t.edge){
 auto&c=caches[{n,h.sid}];
 if(h.txn!=queryTxn[h.sid]||h.version<c.version){Log(n,"stale_reply",h.sid,h.seq,0);continue;}
 c={h.version,h.locator,Now()+ttl};queryUntil[h.sid]=0;Log(n,"mapping_cache",h.sid,h.version,0,h.locator);State();
 std::vector<uint32_t> q;for(auto&v:pending)if(v.first.first==h.sid)q.push_back(v.first.second);
 for(auto seq:q)Try(h.sid,seq);
 }else if(h.type==7&&n==t.edge){
 pending[{h.sid,h.seq}]={born.at({h.sid,h.seq}),p->GetSize(),EventId()};Try(h.sid,h.seq);
 }else if((h.type==4||h.type==5)&&n==t.uav){
 if(h.type==4&&(h.locator!=addresses[n].Get()||h.version!=versions[n])){Log(n,"stale_data",h.sid,h.seq,p->GetSize(),h.locator);continue;}
 if(h.type==4){Ipv4Header ip;p->RemoveHeader(ip);UdpHeader udp;p->RemoveHeader(udp);Message inner;p->RemoveHeader(inner);
 if(ip.GetDestination()!=Ipv4Address(0x0aee0000+t.uav)||ip.GetProtocol()!=17||inner.sid!=h.sid||inner.seq!=h.seq){Log(n,"inner_packet_drop",h.sid,h.seq,0);continue;}
 Log(n,"decap",h.sid,h.seq,p->GetSize(),ip.GetDestination().Get());}
 auto key=std::make_pair(h.sid,h.seq);if(delivered.insert(key).second)Log(n,"complete",h.sid,h.seq,p->GetSize(),Now()-born.at(key));
 h.type=8;Send(n,edgeAddress.at(t.edge),h);
 }else if(h.type==8&&n==t.edge){pending.erase({h.sid,h.seq});}
}}
void Retry(uint32_t sid,uint32_t seq){
 auto it=pending.find({sid,seq});if(it==pending.end())return;
 if(Now()>it->second.born+tasks.at(sid).deadline){Log(tasks.at(sid).edge,"timeout",sid,seq,0);pending.erase(it);return;}
 Try(sid,seq);
}
void Try(uint32_t sid,uint32_t seq){
 auto key=std::make_pair(sid,seq);auto it=pending.find(key);if(it==pending.end())return;const auto&t=tasks.at(sid);
 if(Now()>it->second.born+t.deadline)return;
 it->second.retry.Cancel();
 Mapping m;
 if(t.mode=="id"){
 m=caches[{t.edge,sid}];
 if(m.until<=Now()){
 if(queryUntil[sid]<=Now()){
 Message h;h.type=2;h.sid=sid;h.eid=t.uav;h.seq=seq;h.txn=++queryTxn[sid];queryUntil[sid]=Now()+.2;
 Send(t.edge,edgeAddress.at(t.server),h);Log(t.edge,"query",sid,seq,53,h.txn);
 }
 }else{Message h;h.type=4;h.sid=sid;h.eid=t.uav;h.seq=seq;h.version=m.version;h.locator=m.locator;
 Send(t.edge,Ipv4Address(m.locator),h,it->second.size);Log(t.edge,"encap",sid,seq,it->second.size+90,m.locator);}
 }else{
 auto f=mappings.find({t.server,sid});if(f!=mappings.end()){
 m=f->second;Message h;h.type=5;h.sid=sid;h.eid=t.uav;h.seq=seq;h.version=m.version;h.locator=m.locator;Send(t.edge,Ipv4Address(m.locator),h,it->second.size);
 }}
 it->second.retry=Simulator::Schedule(Seconds(.2),&Retry,sid,seq);
}
class PeerTransport:public nfd::face::Transport{
public:PeerTransport(Ptr<Node> n,Ptr<NetDevice>d,Mac48Address p):node(n),device(d),peer(p){
 std::ostringstream l,r;l<<"netdev://["<<Mac48Address::ConvertFrom(d->GetAddress())<<"]";r<<"netdev://["<<p<<"]";
 setLocalUri(nfd::FaceUri(l.str()));setRemoteUri(nfd::FaceUri(r.str()));setScope(::ndn::nfd::FACE_SCOPE_NON_LOCAL);
 setPersistency(::ndn::nfd::FACE_PERSISTENCY_PERSISTENT);setLinkType(::ndn::nfd::LINK_TYPE_POINT_TO_POINT);setMtu(d->GetMtu());
 n->RegisterProtocolHandler(MakeCallback(&PeerTransport::Rx,this),ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE,d,true);}
 ~PeerTransport()override{node->UnregisterProtocolHandler(MakeCallback(&PeerTransport::Rx,this));}
private:void doClose()override{setState(nfd::face::TransportState::CLOSED);}
 void doSend(const ::ndn::Block&b)override{auto p=Create<Packet>();p->AddHeader(ns3::ndn::BlockHeader(b));device->Send(p,peer,ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE);}
 void Rx(Ptr<NetDevice>d,Ptr<const Packet>p,uint16_t,const Address&f,const Address&t,NetDevice::PacketType k){
 if(d!=device||f!=peer||t!=device->GetAddress()||k==NetDevice::PACKET_OTHERHOST)return;
 auto c=p->Copy();ns3::ndn::BlockHeader h;c->RemoveHeader(h);receive(std::move(h.getBlock()));}
 Ptr<Node>node;Ptr<NetDevice>device;Mac48Address peer;
};
std::shared_ptr<nfd::Face> RadioFace(Ptr<Node>n,Ptr<ns3::ndn::L3Protocol>l3,Ptr<NetDevice>d){
 std::shared_ptr<nfd::Face>first;uint32_t id=n->GetId();
 for(auto&r:radios){uint32_t p=r.first;if((id<5)==(p<5))continue;
 nfd::face::GenericLinkService::Options o;o.allowFragmentation=true;o.allowReassembly=true;
 auto face=std::make_shared<nfd::Face>(std::make_unique<nfd::face::GenericLinkService>(o),std::make_unique<PeerTransport>(n,d,Mac48Address::ConvertFrom(r.second->GetAddress())));
 l3->addFace(face);faces[{id,p}]=face;if(!first)first=face;}return first;
}
class CachePolicy:public nfd::cs::Policy{
public:CachePolicy():Policy("trimodal-leases"){}uint32_t node=0;
private:std::list<EntryRef>queue;
 bool allowed(EntryRef e){for(auto&t:tasks)if(t.second.cache==node&&Now()>=t.second.start&&Now()<=t.second.drain&&e->getName().toUri().find(t.second.prefix+"/")==0)return true;return false;}
 void doAfterInsert(EntryRef e)override{if(!allowed(e)){emitSignal(beforeEvict,e);return;}queue.push_back(e);evictEntries();}
 void doAfterRefresh(EntryRef)override{}void doBeforeUse(EntryRef)override{}
 void doBeforeErase(EntryRef e)override{queue.remove(e);}
 void evictEntries()override{while(getCs()->size()>getLimit()&&!queue.empty()){auto e=queue.front();queue.pop_front();emitSignal(beforeEvict,e);}}
};
class Consumer:public ns3::ndn::App{
public:static TypeId GetTypeId(){static TypeId t=TypeId("ns3::TrimodalConsumer").SetParent<ns3::ndn::App>().AddConstructor<Consumer>();return t;}
 std::map<std::string,std::deque<std::pair<uint32_t,uint32_t>>> waiting;uint32_t nonce=0;
 void Release(uint32_t sid){waiting.erase(tasks.at(sid).prefix+"/v1/chunk0");}
 size_t PendingCount()const{size_t c=0;for(auto&q:waiting)c+=q.second.size();return c;}
 void Emit(uint32_t sid,uint32_t seq){auto name=tasks.at(sid).prefix+"/v1/chunk0";waiting[name].push_back({sid,seq});Tx(sid,seq);}
 void Tx(uint32_t sid,uint32_t seq){
 auto key=std::make_pair(sid,seq);if(delivered.count(key)||Now()>born.at(key)+tasks.at(sid).deadline)return;
 auto p=std::make_shared<::ndn::Interest>(tasks.at(sid).prefix+"/v1/chunk0");p->setNonce(++nonce+GetNode()->GetId()*100000);
 p->setInterestLifetime(::ndn::time::milliseconds(180));p->setMustBeFresh(true);
 m_transmittedInterests(p,this,m_face);m_appLink->onReceiveInterest(*p);Simulator::Schedule(Seconds(.2),&Consumer::Tx,this,sid,seq);
 }
 void OnData(std::shared_ptr<const ::ndn::Data>p)override{
 ns3::ndn::App::OnData(p);auto&q=waiting[p->getName().toUri()];
 // One object may satisfy several concurrent demands for the same immutable version.
 while(!q.empty()){auto k=q.front();q.pop_front();if(Now()>born.at(k)+tasks.at(k.first).deadline)continue;
 if(delivered.insert(k).second)Log(GetNode()->GetId(),"complete",k.first,k.second,p->getContent().value_size(),Now()-born.at(k));}
 }
};
NS_OBJECT_ENSURE_REGISTERED(Consumer);
std::map<uint32_t,Ptr<Consumer>> consumers;
void Fib(uint32_t from,uint32_t to,std::string prefix){
 if(from>=3&&to>=3)ns3::ndn::FibHelper::AddRoute(nodes.Get(from),prefix,faces.at({from,to}),1);
 else ns3::ndn::FibHelper::AddRoute(nodes.Get(from),prefix,nodes.Get(to),1);
}
void Association(uint32_t n,Mac48Address bssid){
 uint32_t ap=0;for(uint32_t p:{3,4})if(radios.at(p)->GetAddress()==bssid)ap=p;
 NS_ABORT_MSG_IF(!ap,"unknown association");attached[n]=ap;++versions[n];
 auto ip=nodes.Get(n)->GetObject<Ipv4>();int iface=ip->GetInterfaceForDevice(radios.at(n));
 while(ip->GetNAddresses(iface))ip->RemoveAddress(iface,0);
 std::ostringstream a;a<<"10.1."<<ap-3<<"."<<n+5;addresses[n]=Ipv4Address(a.str().c_str());
 ip->AddAddress(iface,Ipv4InterfaceAddress(addresses[n],Ipv4Mask("255.255.255.0")));ip->SetUp(iface);
 Ipv4StaticRoutingHelper rh;auto r=rh.GetStaticRouting(ip);while(r->GetNRoutes())r->RemoveRoute(0);
 std::ostringstream gw;gw<<"10.1."<<ap-3<<".1";r->SetDefaultRoute(Ipv4Address(gw.str().c_str()),iface);
 Log(n,"associate",0,versions[n],ap,addresses[n].Get());
 for(auto&t:tasks)if(t.second.uav==n&&Now()>=t.second.start&&Now()<=t.second.drain){
 if(t.second.mode=="ndn"){
 for(uint32_t p:{3,4}){ns3::ndn::FibHelper::RemoveRoute(nodes.Get(n),t.second.prefix,faces.at({n,p}));}Fib(n,ap,t.second.prefix);
 }else Register(t.first,versions[n],0);
 }
}
void Deassociation(uint32_t n,Mac48Address){attached[n]=0;Log(n,"deassociate",0,versions[n],0);}
void Begin(uint32_t sid){const auto&t=tasks.at(sid);Log(t.uav,"slice_start",sid,0,0);
 if(t.mode=="ndn"){Fib(3,1,t.prefix);Fib(4,2,t.prefix);Fib(t.edge,0,t.prefix);Fib(3-t.edge,t.edge,t.prefix);if(attached[t.uav])Fib(t.uav,attached[t.uav],t.prefix);}else if(attached[t.uav])Register(sid,versions[t.uav],0);
}
void End(uint32_t sid){const auto&t=tasks.at(sid);Log(t.uav,"stop_generation",sid,0,0);
 for(uint32_t n:{1,2}){auto&cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n))->getForwarder()->getCs();
 for(auto&e:cs)if(e.getName().toUri().find(t.prefix+"/")==0)Log(n,"cache_entry",sid,0,e.getData().getContent().value_size());}
}
void Release(uint32_t sid){const auto&t=tasks.at(sid);
 queryTxn.erase(sid);queryUntil.erase(sid);registrationAck.erase(sid);consumers.at(t.uav)->Release(sid);
 for(uint32_t n:{1,2}){mappings.erase({n,sid});caches.erase({n,sid});auto&cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n))->getForwarder()->getCs();cs.erase(::ndn::Name(t.prefix),100000,[](size_t){});}
 std::vector<std::pair<uint32_t,uint32_t>>q;for(auto&v:pending)if(v.first.first==sid)q.push_back(v.first);for(auto&k:q)pending.erase(k);
 if(t.mode=="ndn"){
 for(auto&h:hops)ns3::ndn::FibHelper::RemoveRoute(nodes.Get(h.first.first),t.prefix,nodes.Get(h.first.second));
 for(uint32_t ap:{3,4})ns3::ndn::FibHelper::RemoveRoute(nodes.Get(t.uav),t.prefix,faces.at({t.uav,ap}));
 }
 Log(t.uav,"release",sid,0,0);State();
}
void Emit(uint32_t sid,uint32_t seq){const auto&t=tasks.at(sid);born[{sid,seq}]=Now();Log(t.uav,"emit",sid,seq,1000);
 if(t.mode=="ndn")consumers.at(t.uav)->Emit(sid,seq);
 else{Message h;h.type=7;h.sid=sid;h.eid=t.uav;h.seq=seq;Send(0,edgeAddress.at(t.edge),h,1000);}
}
uint32_t NameSlice(const ::ndn::Name& name){for(auto&t:tasks)if(name.toUri().find(t.second.prefix+"/")==0)return t.first;return 0;}
uint32_t Peer(uint32_t n,const nfd::Face&f){
 for(auto&p:faces)if(p.first.first==n&&p.second->getId()==f.getId())return p.first.second;
 auto l3=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n));
 for(auto&h:hops)if(h.first.first==n){auto face=l3->getFaceByNetDevice(h.second.first);if(face&&face->getId()==f.getId())return h.first.second;}
 return 999;
}
void NdnOut(uint32_t n,const ::ndn::Data&p,const nfd::Face&f){Log(n,"ndn_data",NameSlice(p.getName()),f.getId(),p.wireEncode().size(),Peer(n,f));}
void NdnInterest(uint32_t n,const ::ndn::Interest&p,const nfd::Face&f){Log(n,"ndn_interest",NameSlice(p.getName()),f.getId(),p.wireEncode().size(),Peer(n,f));}
void IpTx(uint32_t n,Ptr<const Packet>p,Ptr<Ipv4> ipv4,uint32_t iface){
 auto c=p->Copy();Ipv4Header ip;c->RemoveHeader(ip);if(ip.GetProtocol()!=17)return;UdpHeader udp;c->RemoveHeader(udp);
 if(udp.GetDestinationPort()!=4341||c->GetSize()<9)return;Message h;c->RemoveHeader(h);
 bool local=false;
 for(uint32_t i=0;i<ipv4->GetNInterfaces();++i)for(uint32_t j=0;j<ipv4->GetNAddresses(i);++j)if(ipv4->GetAddress(i,j).GetLocal()==ip.GetDestination())local=true;
 Log(n,(local?"ip_local:":"ip_tx:")+std::to_string(h.type),h.sid,h.seq,p->GetSize(),iface);
}
void WireTx(uint32_t n,uint32_t peer,Ptr<const Packet>p){
 auto c=p->Copy();PppHeader h;c->RemoveHeader(h);
 Log(n,h.GetProtocol()==0x21?"wire_ip":"wire_ndn",0,peer,c->GetSize());
}
void PhyBegin(uint32_t n,Ptr<const Packet>p,double){Log(n,"phy_begin",0,p->GetUid(),p->GetSize());}
void PhyEnd(uint32_t n,Ptr<const Packet>p){Log(n,"phy_end",0,p->GetUid(),p->GetSize());}
void PhyState(uint32_t n,Time,Time duration,WifiPhyState state){if(state==WifiPhyState::TX)Log(n,"phy_tx",0,0,0,duration.GetSeconds());}
}
int main(int argc,char**argv){
 std::string output="trimodal",taskFile;uint32_t seed=101,cacheEntries=16;double backhaulMbps=1.;bool negative=false;
 CommandLine cmd;cmd.AddValue("negative","inject stale and foreign namespace messages",negative);cmd.AddValue("output","output prefix",output);cmd.AddValue("tasks","v1 TSV tasks",taskFile);cmd.AddValue("uavs","UAV count",uavs);
 cmd.AddValue("seed","wireless RNG",seed);cmd.AddValue("speed","moving UAV speed",speed);cmd.AddValue("startX","moving UAV x",startX);
 cmd.AddValue("stop","simulation duration",stopAt);cmd.AddValue("backhaulMbps","wired rate",backhaulMbps);cmd.AddValue("cacheEntries","per-edge CS entries",cacheEntries);cmd.Parse(argc,argv);
 NS_ABORT_MSG_IF(uavs<4||uavs>200,"unsupported UAV count");RngSeedManager::SetSeed(seed);RngSeedManager::SetRun(1);
 events.open(output+"-events.csv");events<<"time,node,event,slice,seq,bytes,detail\n";
 std::ifstream input(taskFile);NS_ABORT_MSG_IF(!input,"missing task file");std::string line;
 while(std::getline(input,line)){if(line.empty()||line[0]=='#')continue;std::istringstream row(line);Task t;
 row>>t.id>>t.tenant>>t.mode>>t.uav>>t.start>>t.end>>t.period>>t.deadline>>t.edge>>t.server>>t.cache;
 NS_ABORT_MSG_IF(!row||t.id==0||tasks.count(t.id)||t.uav<5||t.uav>=5+uavs||t.start<1||t.end<=t.start||t.period<=0||t.deadline<=0||t.edge<1||t.edge>2||t.server<1||t.server>2||(t.mode!="ip"&&t.mode!="id"&&t.mode!="ndn")||(t.cache>2)||t.cache&&t.cache!=t.edge,"invalid task");
 t.drain=t.end+t.deadline+.01;t.prefix="/t"+std::to_string(t.tenant)+"/s"+std::to_string(t.id);tasks[t.id]=t;stopAt=std::max(stopAt,t.drain+1.);
 }
 nodes.Create(5+uavs);MobilityHelper mobility;mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");mobility.Install(nodes);
 std::vector<Vector>positions={Vector(75,-30,0),Vector(0,-10,0),Vector(150,-10,0),Vector(0,0,10),Vector(150,0,10)};
 for(uint32_t i=0;i<uavs;++i)positions.push_back(i==0?Vector(startX,0,80):(i%2?Vector(20,5*(i/2),80):Vector(145,5*(i/2),80)));
 for(uint32_t i=0;i<nodes.GetN();++i)nodes.Get(i)->GetObject<MobilityModel>()->SetPosition(positions[i]);
 nodes.Get(5)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(Vector(speed,0,0));
 InternetStackHelper internet;internet.Install(nodes);Ipv4StaticRoutingHelper rh;
 PointToPointHelper p2p;p2p.SetDeviceAttribute("DataRate",DataRateValue(DataRate(uint64_t(backhaulMbps*1e6))));p2p.SetChannelAttribute("Delay",StringValue("2ms"));
 uint32_t subnet=0;for(auto ab:std::vector<std::pair<uint32_t,uint32_t>>{{0,1},{0,2},{1,2},{1,3},{2,4}}){
 auto d=p2p.Install(nodes.Get(ab.first),nodes.Get(ab.second));Ipv4AddressHelper ah;std::ostringstream net;net<<"10.0."<<subnet++<<".0";ah.SetBase(net.str().c_str(),"255.255.255.0");auto a=ah.Assign(d);
 hops[ab]={d.Get(0),a.GetAddress(1)};hops[{ab.second,ab.first}]={d.Get(1),a.GetAddress(0)};if(ab.first==0)edgeAddress[ab.second]=a.GetAddress(1);
 }
 YansWifiChannelHelper channel;channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");channel.AddPropagationLoss("ns3::RangePropagationLossModel","MaxRange",DoubleValue(120.));
 YansWifiPhyHelper phy;phy.SetChannel(channel.Create());WifiHelper wifi;wifi.SetStandard(WIFI_STANDARD_80211a);
 wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue("OfdmRate6Mbps"),"ControlMode",StringValue("OfdmRate6Mbps"),"NonUnicastMode",StringValue("OfdmRate6Mbps"));
 WifiMacHelper mac;Ssid ssid("lowalt");mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid));
 for(uint32_t n:{3,4}){auto d=wifi.Install(phy,mac,nodes.Get(n));radios[n]=d.Get(0);Ipv4AddressHelper a;std::string net=n==3?"10.1.0.0":"10.1.1.0";a.SetBase(net.c_str(),"255.255.255.0");a.Assign(d);}
 mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false),"MaxMissedBeacons",UintegerValue(3));
 for(uint32_t n=5;n<nodes.GetN();++n){auto d=wifi.Install(phy,mac,nodes.Get(n));radios[n]=d.Get(0);auto ip=nodes.Get(n)->GetObject<Ipv4>();ip->AddInterface(d.Get(0));ip->SetUp(ip->GetInterfaceForDevice(d.Get(0)));
 auto m=DynamicCast<StaWifiMac>(DynamicCast<WifiNetDevice>(d.Get(0))->GetMac());m->TraceConnectWithoutContext("Assoc",MakeBoundCallback(&Association,n));m->TraceConnectWithoutContext("DeAssoc",MakeBoundCallback(&Deassociation,n));}
 // Deterministic ground routing; AP subnets remain distinct when the same UAV moves.
 auto route=[&](uint32_t n,Ipv4Address dst,Ipv4Mask mask,uint32_t next){auto h=hops.at({n,next});auto ip=nodes.Get(n)->GetObject<Ipv4>();rh.GetStaticRouting(ip)->AddNetworkRouteTo(dst,mask,h.second,ip->GetInterfaceForDevice(h.first));};
 for(uint32_t n=0;n<5;++n){for(uint32_t e:{1,2})if(n!=e){uint32_t next=n==0?e:(n>=3?n-2:e);if(n>=3&&next==n)continue;route(n,edgeAddress.at(e),Ipv4Mask("255.255.255.255"),next);}
 for(uint32_t ap:{3,4})if(n!=ap){uint32_t next=n==0?ap-2:(n>=3?n-2:(n==ap-2?ap:ap-2));route(n,Ipv4Address(ap==3?"10.1.0.0":"10.1.1.0"),Ipv4Mask("255.255.255.0"),next);}}
 ns3::ndn::StackHelper stack;stack.AddFaceCreateCallback(WifiNetDevice::GetTypeId(),MakeCallback(&RadioFace));
 for(uint32_t n=0;n<nodes.GetN();++n){stack.setCsSize(n==1||n==2?cacheEntries:0);stack.Install(nodes.Get(n));if(n==1||n==2){auto policy=std::make_unique<CachePolicy>();policy->node=n;auto&cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n))->getForwarder()->getCs();cs.setPolicy(std::move(policy));cs.setLimit(cacheEntries);}}
 for(uint32_t n=0;n<nodes.GetN();++n){
 auto socket=Socket::CreateSocket(nodes.Get(n),UdpSocketFactory::GetTypeId());socket->Bind(InetSocketAddress(Ipv4Address::GetAny(),4341));socket->SetRecvCallback(MakeCallback(&Receive));sockets[n]=socket;
 nodes.Get(n)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext("Tx",MakeBoundCallback(&IpTx,n));
 auto l3=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n));
 l3->TraceConnectWithoutContext("OutData",MakeBoundCallback(&NdnOut,n));
 l3->TraceConnectWithoutContext("OutInterests",MakeBoundCallback(&NdnInterest,n));
 if(n>=5){auto c=CreateObject<Consumer>();nodes.Get(n)->AddApplication(c);c->SetStartTime(Seconds(.1));consumers[n]=c;}
 for(auto&h:hops)if(h.first.first==n){
 h.second.first->TraceConnectWithoutContext("MacTx",MakeBoundCallback(&WireTx,n,h.first.second));
 Log(n,"interface",0,nodes.Get(n)->GetObject<Ipv4>()->GetInterfaceForDevice(h.second.first),h.first.second);
 }
 if(radios.count(n)){auto w=DynamicCast<WifiNetDevice>(radios.at(n));
 w->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeBoundCallback(&PhyBegin,n));
 w->GetPhy()->TraceConnectWithoutContext("PhyTxEnd",MakeBoundCallback(&PhyEnd,n));PointerValue s;w->GetPhy()->GetAttribute("State",s);s.Get<WifiPhyStateHelper>()->TraceConnectWithoutContext("State",MakeBoundCallback(&PhyState,n));}
 }
 ns3::ndn::StrategyChoiceHelper::InstallAll("/","/localhost/nfd/strategy/best-route");
 for(auto&v:tasks){const auto&t=v.second;if(t.mode=="ndn"){
 ns3::ndn::AppHelper producer("ns3::ndn::Producer");producer.SetPrefix(t.prefix);producer.SetAttribute("PayloadSize",UintegerValue(1000));producer.SetAttribute("Freshness",TimeValue(Seconds(60)));producer.Install(nodes.Get(0));}
 Simulator::Schedule(Seconds(t.start),&Begin,t.id);Simulator::Schedule(Seconds(t.end),&End,t.id);Simulator::Schedule(Seconds(t.drain),&Release,t.id);
 uint32_t seq=0;for(double at=t.start+.05;at<t.end-1e-8;at+=t.period)Simulator::Schedule(Seconds(at),&Emit,t.id,++seq);
 }
 if(negative)for(auto&v:tasks)if(v.second.mode=="id"){
 auto t=v.second;
 Simulator::Schedule(Seconds(t.start+1.),[t](){Message h;h.type=1;h.sid=t.id;h.eid=t.uav;h.version=0;Send(0,edgeAddress.at(t.server),h);});
 Simulator::Schedule(Seconds(t.start+1.1),[t](){Message h;h.type=1;h.sid=t.id;h.eid=t.uav+1000;Send(0,edgeAddress.at(t.server),h);});
 Simulator::Schedule(Seconds(t.drain+.2),[t](){Message h;h.type=3;h.sid=t.id;h.eid=t.uav;Send(0,edgeAddress.at(t.edge),h);});
 }
 Simulator::Stop(Seconds(stopAt));Simulator::Run();State();
 size_t remainingApp=0,remainingFib=0;
 for(auto&c:consumers)remainingApp+=c.second->PendingCount();
 for(uint32_t n=1;n<nodes.GetN();++n)for(auto&t:tasks)if(t.second.mode=="ndn"&&ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n))->getForwarder()->getFib().findExactMatch(::ndn::Name(t.second.prefix)))++remainingFib;
 std::ofstream summary(output+"-summary.json");summary<<"{\"schema\":1,\"nodes\":"<<nodes.GetN()<<",\"uavs\":"<<uavs<<",\"generated\":"<<born.size()<<",\"delivered\":"<<delivered.size()<<",\"remaining_mappings\":"<<mappings.size()<<",\"remaining_caches\":"<<caches.size()<<",\"remaining_pending\":"<<pending.size()<<",\"remaining_control\":"<<(queryTxn.size()+queryUntil.size()+registrationAck.size())<<",\"remaining_app_pending\":"<<remainingApp<<",\"remaining_slice_fib\":"<<remainingFib<<"}\n";
 Simulator::Destroy();return 0;
}
